from pickle import dump

from torchvision.models import resnet18

import torch

### USAGE ####
# run this python file to generate snapshot.pickle
# if necessary run: wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/cuda/_memory_viz.py
# run in terminal: python _memory_viz.py memory snapshot.pickle -o memory.svg
# observe the flamegraph!


# needed to generate snapshot
torch.cuda.memory._record_memory_history()

model = resnet18().cuda()
input = torch.rand(1, 3, 224, 224).cuda()
model.train()
snapshot = torch.cuda.memory._snapshot()

# pprint(snapshot['segments'])
dump(snapshot, open("snapshot.pickle", "wb"))
