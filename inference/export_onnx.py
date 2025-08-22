from ultralytics import RTDETR

# Load the pre-trained RT-DETR-L model
model = RTDETR('rtdetr-l.pt')

# Export the model to ONNX format with FP16 precision
# This will create a file named 'rtdetr-l.onnx'
model.export(format='onnx', half=True)

print("Model exported to rtdetr-l.onnx")

