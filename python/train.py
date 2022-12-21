from glob import glob
import sys
from coffea import util
import awkward as ak
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import networks


class ClassInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = ClassInfo(abbreviation='d4', name= 'FourTag Data', color='red')
d3 = ClassInfo(abbreviation='d3', name='ThreeTag Data', color='orange')

def load(cfiles):
    event = []
    for cfile in cfiles:
        print(cfile)
        event.append(util.load(cfile))
    return ak.concatenate(event)

classes = [d4,d3]
for i,c in enumerate(classes): c.index=i

# convert coffea objects in to pytorch tensors
train_valid_modulus = 3
def coffea_to_tensor(event):
    J = torch.FloatTensor( event['Jet',('pt','eta','phi','mass')].to_numpy().view(np.float32).reshape(-1,4,4) ) # [event,jet,feature]
    J = J.transpose(1,2).contiguous() # [event,feature,jet]
    w = torch.FloatTensor( event['weight'].to_numpy().view(np.float32) )
    y = torch.LongTensor( np.asarray(event['class'], dtype=np.uint8) )
    R  = 1*torch.LongTensor( event['SB'].to_numpy().view(np.uint8) )
    R += 2*torch.LongTensor( event['SR'].to_numpy().view(np.uint8) )
    e = torch.LongTensor( np.asarray(event['event'], dtype=np.uint8) )%train_valid_modulus
    dataset = TensorDataset(J, y, w, R, e)
    return dataset


lr_init  = 0.01
lr_scale = 0.25
bs_milestones = [1,3,6,10]
lr_milestones = [15,16,17,18,19,20,21,22,23,24]

train_batch_size = 2**10
infer_batch_size = 2**14
max_train_batch_size = train_batch_size*64


class Loader_Result:
    def __init__(self, trainer, dataset, n_classes=2, train=False):
        self.infer_loader = DataLoader(dataset=dataset, batch_size=infer_batch_size, shuffle=False, num_workers=1, pin_memory=True)
        self.train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True,  num_workers=1, pin_memory=True, drop_last=True) if train else None
        self.n = len(dataset)
        self.w = dataset.tensors[2]
        self.w_sum = self.w.sum()
        self.cross_entropy = torch.zeros(self.n)
        self.n_done = 0
        self.loaded_die_loss = trainer.loaded_die_loss

    def eval(self):
        self.n_done = 0
        
    def infer_batch(self, c_logits, q_logits, y, w, R, e):
        n_batch = c_logits.shape[0]
        y_pred_batch = F.softmax(c_logits, dim=-1)
        cross_entropy_batch = F.cross_entropy(c_logits, y, reduction='none')

        self.cross_entropy[self.n_done:self.n_done+n_batch] = cross_entropy_batch

        self.n_done += n_batch

    def infer_done(self):
        self.loss = (self.w * self.cross_entropy).sum()/self.w_sum/self.loaded_die_loss
        self.n_done = 0


print_step = 10
class Trainer:
    def __init__(self, train_valid_offset=0):
        self.train_valid_offset = train_valid_offset
        self.network = networks.Basic_CNN(8)
        n_trainable_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f'Network has {n_trainable_parameters} trainable parameters')

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr_init, amsgrad=False)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma=lr_scale, last_epoch=-1)

        
    def make_loaders(self, event):
        wd4 = sum(event[event.d4].weight)
        wd3 = sum(event[event.d3].weight)
        w   = sum(event.weight)
        fC = torch.FloatTensor([wd4/w, wd3/w])
        # compute the loss you would get if you only used the class fraction to predict class probability (ie an n_class sided die loaded to land with the right fraction on each class)
        self.loaded_die_loss = -(fC*fC.log()).sum()

        #Split into training and validaiton sets and format into pytorch tensors
        valid = event.event%train_valid_modulus == self.train_valid_offset
        train = ~valid

        dataset_train = coffea_to_tensor(event[train])
        dataset_valid = coffea_to_tensor(event[valid])

        self.train_result = Loader_Result(self, dataset_train, train=True)
        self.valid_result = Loader_Result(self, dataset_valid)

        print(f'{self.train_result.n:,} training samples split into {len(self.train_result.train_loader):,} batches of {train_batch_size:,}')


    @torch.no_grad()
    def inference(self, result):
        self.network.eval()
        result.eval()

        n_done = 0
        for i, (J, y, w, R, e) in enumerate(result.infer_loader):
            c_logits, q_logits = self.network(J)
            result.infer_batch(c_logits, q_logits, y, w, R, e)

            if int(i+1) % print_step == 0:
                percent = float(i+1)*100/len(result.infer_loader)
                sys.stdout.write(f'\rEvaluating {percent:3.0f}%')
                sys.stdout.flush()

        result.infer_done()

    def train_inference(self):
        self.inference(self.train_result)
        sys.stdout.write(' '*200)
        sys.stdout.flush()
        print('\r',end='')
        print(f'{self.train_result.loss}')

if __name__ == '__main__':
    #Load data
    coffea_4b = sorted(glob('data/fourTag_picoAOD*.coffea'))
    coffea_3b = sorted(glob('data/threeTag_picoAOD*.coffea'))

    event_3b = load(coffea_3b)
    event_4b = load(coffea_4b)

    event_3b['d3'] = True
    event_3b['d4'] = False

    event_4b['d3'] = False
    event_4b['d4'] = True
    
    event = ak.concatenate([event_3b, event_4b])

    event['class'] = d4.index*event.d4 + d3.index*event.d3 # for binary classification this seems dumb, makes sense when you have multi-class classification

    t=Trainer()
    t.make_loaders(event)
    t.train_inference()


    
