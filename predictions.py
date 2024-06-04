import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import pudb
from PIL import Image

model = torchvision.models.resnet50(pretrained=False)
#model.load_state_dict(torch.load('model.pth'))

def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        return setattr(obj, names[0], val)
    else:
        return set_attr(getattr(obj, names[0]), names[1:], val)

loaded = torch.load("model.pth")
sized_mismatched_keys = []
for k, v in model.state_dict().items():
    if v.size() != loaded['model'][k].size():
        a = set_attr(model, k.split("."), None)
        sized_mismatched_keys.append(k)

model.load_state_dict(loaded['model'], strict=False)
for missed_key in sized_mismatched_keys:
    set_attr(model, missed_key.split("."), torch.nn.Parameter(loaded['model'][missed_key]).to(device))


current_model_dict = model.state_dict()
loaded_state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
model.load_state_dict(new_state_dict, strict=False)

print(model)