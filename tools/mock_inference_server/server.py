import grpc
from concurrent import futures
from tritonclient.grpc import service_pb2, service_pb2_grpc
import numpy as np

class MockInferenceServer(service_pb2_grpc.GRPCInferenceServiceServicer):
    def ModelInfer(self, request, context):
        # Mock bounding box response as bytes
        mock_data = b'[{"label": "button", "coords": [100, 100, 150, 150], "confidence": 0.95}]'
        output = service_pb2.ModelInferResponse.InferOutputTensor(
            name="boxes",
            datatype="BYTES",
            shape=[1],
            contents=service_pb2.ModelInferResponse.InferTensorContents(
                bytes_contents=service_pb2.ModelInferResponse.InferTensorContents.BytesContents(values=[mock_data])
            )
        )
        return service_pb2.ModelInferResponse(outputs=[output])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(MockInferenceServer(), server)
    server.add_insecure_port('[::]:8001')
    server.start()
    print("Mock Triton server running on port 8001")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
