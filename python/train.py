from glob import glob
from coffea import util
import awkward as ak
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class classInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = classInfo(abbreviation='d4', name= 'FourTag Data', color='red')
d3 = classInfo(abbreviation='d3', name='ThreeTag Data', color='orange')

#Load data
coffea_4b = sorted(glob('data/fourTag_picoAOD*.coffea'))
coffea_3b = sorted(glob('data/threeTag_picoAOD*.coffea'))

def load(cfiles):
    event = []
    for cfile in cfiles:
        print(cfile)
        event.append(util.load(cfile))
    return ak.concatenate(event)

event_3b = load(coffea_3b)
event_4b = load(coffea_4b)

event_3b['d3'] = True
event_3b['d4'] = False

event_4b['d3'] = False
event_4b['d4'] = True

event = ak.concatenate([event_3b, event_4b])

classes = [d4,d3]
for i,c in enumerate(classes): c.index=i

event['class'] = d4.index*event.d4 + d3.index*event.d3 # for binary classification this seems dumb, makes sense when you have multi-class classification

#Split into training and validaiton sets and format into pytorch tensors
train_valid_modulus = 3
train_valid_offset  = 0

event['valid'] = event.event%train_valid_modulus == train_valid_offset
event['train'] = ~event.valid

# convert coffea objects in to pytorch tensors
def coffea_to_tensor(event):
    J = torch.FloatTensor( event['Jet',('pt','eta','phi','mass')].to_numpy().view(np.float32).reshape(-1,4,4) ) # [event,jet,feature]
    w = torch.FloatTensor( event['weight'].to_numpy().view(np.float32) )
    y = torch.LongTensor( np.asarray(event['class'], dtype=np.uint8) )
    R  = 1*torch.LongTensor( event['SB'].to_numpy().view(np.uint8) )
    R += 2*torch.LongTensor( event['SR'].to_numpy().view(np.uint8) )
    e = torch.LongTensor( np.asarray(event['event'], dtype=np.uint8) )%train_valid_modulus
    dataset = TensorDataset(J, y, w, R, e)
    return dataset

dataset_train = coffea_to_tensor(event[event.train])
dataset_valid = coffea_to_tensor(event[event.valid])

