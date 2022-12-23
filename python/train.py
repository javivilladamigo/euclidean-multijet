from glob import glob
import sys, os, argparse, re
from copy import copy, deepcopy
from coffea import util
import awkward as ak
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import networks

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #this doesn't work, need to run `conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1` and then reactivate the conda environment

np.random.seed(0)
torch.manual_seed(0)#make training results repeatable 

SCREEN_WIDTH = 100

class ClassInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = ClassInfo(abbreviation='d4', name= 'FourTag Data', color='red')
d3 = ClassInfo(abbreviation='d3', name='ThreeTag Data', color='orange')

def load(cfiles, selection=''):
    event_list = []
    for cfile in cfiles:
        print(cfile,selection)
        event = util.load(cfile)
        if selection:
            event = event[eval(selection)]
        event_list.append(event)
    return ak.concatenate(event_list)

classes = [d4,d3]
for i,c in enumerate(classes): c.index=i

# convert coffea objects in to pytorch tensors
train_valid_modulus = 3
def coffea_to_tensor(event, device='cpu', kfold=False):
    j = torch.FloatTensor( event['Jet',('pt','eta','phi','mass')].to_numpy().view(np.float32).reshape(-1,4,4) ) # [event,jet,feature]
    j = j.transpose(1,2).contiguous() # [event,feature,jet]
    e = torch.LongTensor( np.asarray(event['event'], dtype=np.uint8) )%train_valid_modulus
    if kfold:
        return j, e
    w = torch.FloatTensor( event['weight'].to_numpy().view(np.float32) )
    y = torch.LongTensor( np.asarray(event['class'], dtype=np.uint8) )
    R  = 1*torch.LongTensor( event['SB'].to_numpy().view(np.uint8) )
    R += 2*torch.LongTensor( event['SR'].to_numpy().view(np.uint8) )
    if device != 'cpu':
        j, y, w, R, e = j.to(device), y.to(device), w.to(device), R.to(device), e.to(device)
    dataset = TensorDataset(j, y, w, R, e)
    return dataset


lr_init  = 0.01
lr_scale = 0.25
bs_scale = 2
bs_milestones = [1,3,6,10]
lr_milestones = [15,16,17,18,19,20,21,22,23,24]

train_batch_size = 2**10
infer_batch_size = 2**14
max_train_batch_size = train_batch_size*64

num_workers=8

class Loader_Result:
    def __init__(self, model, dataset, n_classes=2, train=False):
        self.dataset = dataset
        self.infer_loader = DataLoader(dataset=dataset, batch_size=infer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        self.train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True) if train else None
        self.n = len(dataset)
        self.w = dataset.tensors[2]
        self.w_sum = self.w.sum()
        self.cross_entropy = torch.zeros(self.n)
        self.n_done = 0
        self.loaded_die_loss = model.loaded_die_loss
        self.loss_estimate = 1.0
        self.history = {'loss': []}
        self.task = model.task

    def eval(self):
        self.n_done = 0
        
    def infer_batch(self, c_logits, q_logits, y, w, R, e):
        n_batch = c_logits.shape[0]
        # y_pred_batch = F.softmax(c_logits, dim=-1)
        cross_entropy_batch = F.cross_entropy(c_logits, y, reduction='none')

        self.cross_entropy[self.n_done:self.n_done+n_batch] = cross_entropy_batch

        self.n_done += n_batch

    def infer_done(self):
        self.loss = (self.w * self.cross_entropy).sum()/self.w_sum/self.loaded_die_loss
        self.history['loss'].append(copy(self.loss))
        self.n_done = 0

    def train_batch(self, c_logits, q_logits, y, w, R, e):
        if self.task=='FvT':
            isSB = (R==1) # mask SR events in training, we can keep them in inference for now, though with real data this would be a bad idea
            w_isSB = w[isSB]
            w_isSB_sum = w_isSB.sum()
            cross_entropy_batch = F.cross_entropy(c_logits[isSB], y[isSB], reduction='none')
            loss_batch = (w_isSB * cross_entropy_batch).sum()/w_isSB_sum/self.loaded_die_loss
            
        loss_batch.backward()
        self.loss_estimate = self.loss_estimate*0.98 + loss_batch.item()*(1-0.98) # running average with 0.98 exponential decay rate


class Model:
    def __init__(self, train_valid_offset=0, device='cpu', task='FvT', model_file=''): # FvT == Four vs Threetag classification
        self.task = task
        self.device = device
        self.train_valid_offset = train_valid_offset
        self.network = networks.Basic_CNN(8)
        self.network.to(self.device)
        n_trainable_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f'Network has {n_trainable_parameters} trainable parameters')
        self.epoch = 0
        self.lr_current = lr_init
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr_init, amsgrad=False)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma=lr_scale, last_epoch=-1)
        self.lr_change = []
        self.bs_change = []

        if model_file:
            print(f'Load {model_file}')
            model_name_parts = re.split(r'[/_]', model_file)
            self.task = model_name_parts[1]
            self.train_valid_offset = model_name_parts[model_name_parts.index('offset')+1]
            self.model_pkl = model_file
            self.model_dict = torch.load(self.model_pkl)
            self.network.load_state_dict(self.model_dict['model'])
            self.optimizer.load_state_dict(self.model_dict['optimizer'])
            self.epoch = self.model_dict['epoch']
        
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

        dataset_train = coffea_to_tensor(event[train], device=self.device)
        dataset_valid = coffea_to_tensor(event[valid], device=self.device)

        self.train_result = Loader_Result(self, dataset_train, train=True)
        self.valid_result = Loader_Result(self, dataset_valid)

        print(f'{self.train_result.n:,} training samples split into {len(self.train_result.train_loader):,} batches of {train_batch_size:,}')


    @torch.no_grad()
    def inference(self, result):
        self.network.eval()

        for i, (j, y, w, R, e) in enumerate(result.infer_loader):
            c_logits, q_logits = self.network(j)
            result.infer_batch(c_logits, q_logits, y, w, R, e)

            percent = float(i+1)*100/len(result.infer_loader)
            sys.stdout.write(f'\rEvaluating {percent:3.0f}%')
            sys.stdout.flush()

        result.infer_done()

    def train_inference(self):
        self.inference(self.train_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'Epoch {self.epoch:>2} | Training   | Loss {self.train_result.loss:1.5}')

    def valid_inference(self):
        self.inference(self.valid_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'         | Validation | Loss {self.valid_result.loss:1.5}')

    def train(self, result=None):
        if result is None: result = self.train_result
        self.network.train()

        for i, (j, y, w, R, e) in enumerate(result.train_loader):
            self.optimizer.zero_grad()
            c_logits, q_logits = self.network(j)
            result.train_batch(c_logits, q_logits, y, w, R, e)

            percent = float(i+1)*100/len(result.train_loader)
            sys.stdout.write(f'\rTraining {percent:3.0f}% >>> Loss Estimate {result.loss_estimate:1.5f}')
            sys.stdout.flush()

            self.optimizer.step()

    def increment_train_loader(self, new_batch_size=None):
        current_batch_size = self.train_result.train_loader.batch_size
        if new_batch_size is None: new_batch_size = current_batch_size*bs_scale
        if new_batch_size == current_batch_size: return
        print(f'Change training batch size: {current_batch_size} -> {new_batch_size} ({self.train_result.n//new_batch_size} batches)')
        del self.train_result.train_loader
        self.train_result.train_loader = DataLoader(dataset=self.train_result.dataset, batch_size=new_batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
        self.bs_change.append(self.epoch)
            
    def run_epoch(self):
        self.epoch += 1
        self.train()
        self.train_inference()
        self.valid_inference()
        self.scheduler.step()
        if (self.epoch in bs_milestones or self.epoch in lr_milestones) and self.network.n_ghost_batches:
            gb_decay = 4 #2 if self.epoch in bs_mile
            print(f'set_ghost_batches({self.network.n_ghost_batches//gb_decay})')
            self.network.set_ghost_batches(self.network.n_ghost_batches//gb_decay)
        if self.epoch in bs_milestones:
            self.increment_train_loader()
        if self.epoch in lr_milestones:
            print(f'Decay learning rate: {self.lr_current} -> {self.lr_current*lr_scale}')
            self.lr_current *= lr_scale
            self.lr_change.append(self.epoch)

    def run_training(self):
        self.network.set_mean_std(self.train_result.dataset.tensors[0])
        self.train_inference()
        self.valid_inference()
        for _ in range(20):
            self.run_epoch()
        self.save_model()

    def save_model(self):
        self.history = {'train': self.train_result.history,
                        'valid': self.valid_result.history}
        self.model_dict = {'model': deepcopy(self.network.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict()),
                           'epoch': self.epoch,
                           'history': copy(self.history)}
        self.model_pkl = f'models/{self.task}_{self.network.name}_offset_{self.train_valid_offset}_epoch_{self.epoch:02d}.pkl'
        print(self.model_pkl)
        torch.save(self.model_dict, self.model_pkl)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default=False, action='store_true', help='Run model training')
    parser.add_argument('-o', '--offset', default=0, type=int, help='k-folding offset for split between training/validation sets')
    parser.add_argument('-m', '--model', default='', type=str, help='Load these models (* wildcard for offsets)')
    args = parser.parse_args()

    if args.train:
        task = 'FvT'
        #Load data
        if task == 'FvT':
            coffea_4b = sorted(glob('data/fourTag_picoAOD*.coffea'))
            coffea_3b = sorted(glob('data/threeTag_picoAOD*.coffea'))

            event_3b = load(coffea_3b, selection='event.preselection & event.SB')
            event_4b = load(coffea_4b, selection='event.preselection & event.SB')

            event_3b['d3'] = True
            event_3b['d4'] = False

            event_4b['d3'] = False
            event_4b['d4'] = True

            event = ak.concatenate([event_3b, event_4b])

            event['class'] = d4.index*event.d4 + d3.index*event.d3 # for binary classification this seems dumb, makes sense when you have multi-class classification

        model_args = {'task': task,
                      'train_valid_offset': args.offset}
        t=Model(**model_args) # Four vs Threetag classification
        t.make_loaders(event)
        t.run_training()

    if args.model:
        model_files = sorted(glob(args.model))
        models = []
        for model_file in model_files:
            models.append(Model(model_file=model_file))

        task = models[0].task
        kfold = networks.K_Fold([model.network for model in models])

        import uproot
        import awkward as ak

        picoAODs = glob('data/*picoAOD.root')
        for picoAOD in picoAODs:
            output_file = picoAOD.replace('picoAOD', task)
            print(f'Generate kfold output for {picoAOD} -> {output_file}')
            coffea_files = sorted(glob(picoAOD.replace('.root','*.coffea')))
            event = load(coffea_files)
            j, e = coffea_to_tensor(event, kfold=True)
            c_logits, q_logits = kfold(j, e)
            c_score, q_score = F.softmax(c_logits, dim=1), F.softmax(q_logits, dim=1)

            with uproot.recreate(output_file) as output:
                kfold_dict = {}
                kfold_dict['q_0123'] = q_score[:,0].numpy()
                kfold_dict['q_0213'] = q_score[:,1].numpy()
                kfold_dict['q_0312'] = q_score[:,2].numpy()
                for cl in classes:
                    kfold_dict[cl.abbreviation] = c_score[:,cl.index].numpy()
                if task == 'FvT':
                    kfold_dict['rw'] = kfold_dict['d4'] / kfold_dict['d3']
                output['Events'] = {task: ak.zip(kfold_dict),
                                    'event': event.event}
            
