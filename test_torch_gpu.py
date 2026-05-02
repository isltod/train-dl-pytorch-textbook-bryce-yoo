import torch
print(f"CUDA 가용 여부: {torch.cuda.is_available()}")
print(f"GPU 장치 수: {torch.cuda.device_count()}")
print(f"GPU 0 Name: {torch.cuda.get_device_name(0)}")
print(f"GPU 1 Name: {torch.cuda.get_device_name(1)}")
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
