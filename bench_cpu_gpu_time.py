import time

import torch

device = "cuda"
bytes = 256400000  # 256400000
pin = False
gpu_loaded = False


if gpu_loaded:
    num_elements = int(15e9 // torch.LongTensor().element_size())
    load = torch.LongTensor(num_elements)
    load = load.to(device)

num_elements = bytes // torch.LongTensor().element_size()
t = torch.LongTensor(num_elements)

if pin:
    t = t.pin_memory()

start = time.time()
t = t.to(device)
end = time.time()
duration = end - start
print(f"Duration is {duration}")
bd = bytes / duration
print(f"Bandwidth CPU -> GPU is {bd/1e6}  MB")
print(f"Device {t.device}")
