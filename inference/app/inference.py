import os
import threading
import tensorrt as trt
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Tuple, Union

def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL:  torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]

def _dims_to_tuple(dims) -> Tuple[int, ...]:
    try:
        return tuple(int(d) for d in dims)
    except Exception:
        return tuple(dims)

class RTDETREngine:
    """
    TRT wrapper supporting legacy(v2) and tensors(v3) APIs.

    Handles outputs in any of these shapes:
      A) [1, N, 6] or [1, 6, N]    -> (cx,cy,w,h,class_id,score)
      B) [1, N, 4+C] or [1, 4+C, N]-> (cx,cy,w,h, class_logits[C])  -> argmax
      C) two outputs: boxes [1,N,4]/[1,4,N] + logits [1,N,C]/[1,C,N]
    """
    def __init__(self, engine_path: str | None = None, conf_threshold: float = 0.5):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.conf_threshold = conf_threshold
        self.lock = threading.Lock()               # serialize context use
        self.stream = torch.cuda.Stream()          # non-default CUDA stream for v3

        if engine_path is None:
            engine_path = os.environ.get("ENGINE_PATH", "/app/models/rtdetr-l.engine")
        self.engine_path = engine_path

        try:
            with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Error loading TensorRT engine from {self.engine_path}: {e}")
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # API flavor detection
        self.use_v3 = hasattr(self.engine, "num_io_tensors") and callable(getattr(self.engine, "get_tensor_name", None))

        if self.use_v3:
            self.all_tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.all_tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(self.input_names) != 1:
                raise ValueError(f"Expected exactly 1 input, found: {self.input_names}")
            self.output_tensors: Dict[str, Tuple[torch.Tensor, Tuple[int, ...]]] = {}
        else:
            if not hasattr(self.engine, "num_bindings"):
                raise RuntimeError("Neither v3 tensors API nor legacy num_bindings found.")
            self.num_bindings = self.engine.num_bindings
            self.input_indices = [i for i in range(self.num_bindings) if self.engine.binding_is_input(i)]
            self.output_indices = [i for i in range(self.num_bindings) if not self.engine.binding_is_input(i)]
            if len(self.input_indices) != 1:
                raise ValueError(f"Expected exactly 1 input, got {len(self.input_indices)}.")
            self.bindings: List[int] = [0] * self.num_bindings
            self.output_tensors: Dict[int, Tuple[torch.Tensor, Tuple[int, ...]]] = {}

    # ---------- Pre/Post ----------
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize->RGB->[0,1]->CHW->NCHW; return CUDA tensor (float32 by default)."""
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        tensor = torch.from_numpy(img).unsqueeze(0).contiguous()  # [1,3,640,640]
        return tensor.cuda(non_blocking=True)

    def _cast_input_to_expected(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.use_v3:
            expected = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.input_names[0]))
        else:
            expected = _trt_dtype_to_torch(self.engine.get_binding_dtype(self.input_indices[0]))
        if input_tensor.dtype != expected:
            if expected in (torch.float16, torch.float32):
                input_tensor = input_tensor.to(expected)
            else:
                raise TypeError(f"Engine expects input dtype {expected}, unsupported cast from {input_tensor.dtype}")
        return input_tensor

    # ---------- Inference binding setup ----------
    def _setup_bindings(self, input_tensor: torch.Tensor) -> None:
        input_tensor = self._cast_input_to_expected(input_tensor)
        inp_idx = self.input_indices[0]
        self.context.set_binding_shape(inp_idx, tuple(input_tensor.shape))
        self.bindings = [0] * self.num_bindings
        self.bindings[inp_idx] = int(input_tensor.data_ptr())

        self.output_tensors = {}
        for out_idx in self.output_indices:
            out_dtype = _trt_dtype_to_torch(self.engine.get_binding_dtype(out_idx))
            out_shape = _dims_to_tuple(self.context.get_binding_shape(out_idx))
            if any(d < 0 for d in out_shape):
                raise RuntimeError(f"Unresolved dynamic output shape for binding {out_idx}: {out_shape}")
            numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
            out_buf = torch.empty(numel, dtype=out_dtype, device='cuda')
            self.bindings[out_idx] = int(out_buf.data_ptr())
            self.output_tensors[out_idx] = (out_buf, out_shape)

    def _setup_v3_tensors(self, input_tensor: torch.Tensor) -> None:
        input_tensor = self._cast_input_to_expected(input_tensor)
        inp_name = self.input_names[0]
        self.context.set_input_shape(inp_name, tuple(input_tensor.shape))
        self.context.set_tensor_address(inp_name, int(input_tensor.data_ptr()))

        # Allocate outputs
        self.output_tensors = {}
        for name in self.output_names:
            out_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            out_shape = _dims_to_tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in out_shape):
                raise RuntimeError(f"Unresolved dynamic output shape for tensor '{name}': {out_shape}")
            numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
            out_buf = torch.empty(numel, dtype=out_dtype, device='cuda')
            self.context.set_tensor_address(name, int(out_buf.data_ptr()))
            self.output_tensors[name] = (out_buf, out_shape)

    # ---------- Output collection ----------
    @staticmethod
    def _reshape_io(io_dict: Dict[Union[str, int], Tuple[torch.Tensor, Tuple[int, ...]]]) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for _, (buf, shape) in io_dict.items():
            outs.append(buf.reshape(shape))
        return outs

    # ---------- Postprocess variants ----------
    def _post_from_single_tensor(self, out: torch.Tensor, original_hw: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Handle [1,N,6] / [1,6,N] OR [1,N,4+C] / [1,4+C,N].
        """
        h, w = original_hw

        if out.ndim != 3:
            raise ValueError(f"Unexpected output ndim: {out.ndim}")

        # Normalize to [N, K]
        if out.shape[0] != 1:
            raise ValueError(f"Batch > 1 not supported; got {tuple(out.shape)}")
        if out.shape[2] >= out.shape[1]:
            arr = out[0]                # [N, K]
        else:
            arr = out[0].transpose(0,1) # [N, K]

        N, K = arr.shape
        detections: List[Dict[str, Any]] = []

        if K == 6:
            # (cx,cy,w,h, cls_id, score) — your original assumption
            dets = arr.float().detach().cpu().numpy()
            for cx, cy, bw, bh, cls_id, score in dets:
                if score < self.conf_threshold:
                    continue
                x1 = (cx - bw/2.0) * w;  y1 = (cy - bh/2.0) * h
                x2 = (cx + bw/2.0) * w;  y2 = (cy + bh/2.0) * h
                detections.append({
                    "box": [float(np.clip(x1,0,w-1)), float(np.clip(y1,0,h-1)),
                            float(np.clip(x2,0,w-1)), float(np.clip(y2,0,h-1))],
                    "score": float(score),
                    "label": int(cls_id),
                })
            return detections

        if K >= 5:
            # (cx,cy,w,h) + class logits (C = K-4)
            boxes = arr[:, :4]          # [N,4]
            logits = arr[:, 4:]         # [N,C]
            # DETR-style heads are typically softmax; if your export used sigmoid, this still works reasonably.
            probs = torch.softmax(logits, dim=-1)
            scores, labels = torch.max(probs, dim=-1)  # [N], [N]

            boxes = boxes.float().detach().cpu().numpy()
            scores = scores.float().detach().cpu().numpy()
            labels = labels.int().detach().cpu().numpy()

            for i in range(N):
                score = float(scores[i])
                if score < self.conf_threshold:
                    continue
                cx, cy, bw, bh = boxes[i].tolist()
                x1 = (cx - bw/2.0) * w;  y1 = (cy - bh/2.0) * h
                x2 = (cx + bw/2.0) * w;  y2 = (cy + bh/2.0) * h
                detections.append({
                    "box": [float(np.clip(x1,0,w-1)), float(np.clip(y1,0,h-1)),
                            float(np.clip(x2,0,w-1)), float(np.clip(y2,0,h-1))],
                    "score": score,
                    "label": int(labels[i]),
                })
            return detections

        raise ValueError(f"Unexpected K={K} in output of shape {tuple(out.shape)}")

    def _post_from_boxes_logits(self, boxes_t: torch.Tensor, logits_t: torch.Tensor,
                                original_hw: Tuple[int,int]) -> List[Dict[str, Any]]:
        """Handle two outputs: boxes [1,N,4]/[1,4,N] and logits [1,N,C]/[1,C,N]."""
        h, w = original_hw

        def to_NXK(t: torch.Tensor) -> torch.Tensor:
            assert t.ndim == 3 and t.shape[0] == 1
            return t[0] if t.shape[2] >= t.shape[1] else t[0].transpose(0,1)

        boxes = to_NXK(boxes_t)   # [N,4]
        logits = to_NXK(logits_t) # [N,C]

        probs = torch.softmax(logits, dim=-1)
        scores, labels = torch.max(probs, dim=-1)

        boxes = boxes.float().detach().cpu().numpy()
        scores = scores.float().detach().cpu().numpy()
        labels = labels.int().detach().cpu().numpy()

        detections: List[Dict[str, Any]] = []
        for i, (cx, cy, bw, bh) in enumerate(boxes):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue
            x1 = (cx - bw/2.0) * w;  y1 = (cy - bh/2.0) * h
            x2 = (cx + bw/2.0) * w;  y2 = (cy + bh/2.0) * h
            detections.append({
                "box": [float(np.clip(x1,0,w-1)), float(np.clip(y1,0,h-1)),
                        float(np.clip(x2,0,w-1)), float(np.clip(y2,0,h-1))],
                "score": score,
                "label": int(labels[i]),
            })
        return detections

    # ---------- Public detect ----------
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        with self.lock:  # serialize access to a single TRT context
            original_hw = image.shape[:2]
            input_tensor = self._preprocess(image)

            if self.use_v3:
                self._setup_v3_tensors(input_tensor)

                # Ensure input copy to GPU is complete before launching on our non-default stream
                torch.cuda.synchronize()
                ok = self.context.execute_async_v3(self.stream.cuda_stream)
                self.stream.synchronize()
                if not ok:
                    raise RuntimeError("TensorRT v3 execution failed.")

                outs = self._reshape_io(self.output_tensors)
            else:
                self._setup_bindings(input_tensor)
                ok = self.context.execute_v2(self.bindings)
                if not ok:
                    raise RuntimeError("TensorRT v2 execution failed.")
                outs = self._reshape_io(self.output_tensors)

            # Postprocess depending on how many outputs the engine produced
            if len(outs) == 1:
                return self._post_from_single_tensor(outs[0], original_hw)

            # Try boxes+logits pairing
            boxes_t, logits_t = None, None
            for t in outs:
                shp = tuple(t.shape)
                if 4 in shp and shp.count(4) == 1 and max(shp) > 4:
                    # heuristically treat the [*,*,4] tensor as boxes
                    boxes_t = t if boxes_t is None else boxes_t
                else:
                    logits_t = t if logits_t is None else logits_t

            if boxes_t is not None and logits_t is not None:
                return self._post_from_boxes_logits(boxes_t, logits_t, original_hw)

            # Fallback: just attempt single-tensor path on the first output
            return self._post_from_single_tensor(outs[0], original_hw)
