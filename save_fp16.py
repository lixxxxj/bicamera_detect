import onnx
from onnxconverter_common import float16

model = onnx.load(R'D:\study\visualstudio\ultralytics-main\examples\YOLOv8-ONNXRuntime-CPP\yolov8n.onnx')
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, R'yolov8n_fp16.onnx')