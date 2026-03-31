# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 06:25:18 2024

@author: Yoush
"""

from ultralytics import YOLO


model_path = 'C:/Users/Yoush/Downloads/Yolov11_Swin/ultralytics-main/runs/detect/train/weights/best.pt'
# Load a COCO-pretrained YOLO11n model
model = YOLO(model_path)

model.export(format = 'onnx')
  
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat

model_fp32 = 'C:/Users/Yoush/Downloads/Yolov11_Swin/ultralytics-main/runs/detect/train/weights/pre.onnx'
model_int8 = 'C:/Users/Yoush/Downloads/Yolov11_Swin/ultralytics-main/runs/detect/train/weights/dynamic_quantized.onnx'

# Quantize 
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8)

# Use the calibration_data_reader with quantize_static
quantize_static('models/pre.onnx', "models/static_quantized.onnx",
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibration_data_reader=calibration_data_reader,
                quant_format=QuantFormat.QDQ,
                nodes_to_exclude=['/model.22/Concat_3', '/model.22/Split', '/model.22/Sigmoid',
                                 '/model.22/dfl/Reshape', '/model.22/dfl/Transpose', '/model.22/dfl/Softmax', 
                                 '/model.22/dfl/conv/Conv', '/model.22/dfl/Reshape_1', '/model.22/Slice_1',
                                 '/model.22/Slice', '/model.22/Add_1', '/model.22/Sub', '/model.22/Div_1',
                                  '/model.22/Concat_4', '/model.22/Mul_2', '/model.22/Concat_5'],
                per_channel=False,
                reduce_range=True,)