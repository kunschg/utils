import time

import torch

device = "cuda"
bytes = 256400  # 256400
pin = True
nb_tensors = 1
gpu_loaded = False


if gpu_loaded:
    num_elements = int(15e9 // torch.LongTensor().element_size())
    load = torch.LongTensor(num_elements)
    load = load.to(device)

tensors_list = []
num_elements = bytes // torch.LongTensor().element_size()
for _ in range(nb_tensors):
    tensors_list.append(torch.LongTensor(int(num_elements // nb_tensors)))

if pin:
    for t in tensors_list:
        t = t.pin_memory()

start = time.time()
for t in tensors_list:
    t = t.to(device)
end = time.time()
duration = end - start
print(f"Duration is {duration}")
bd = bytes / duration
print(f"Bandwidth CPU -> GPU is {bd/1e6}  MB")
print(f"Device {t.device}")
