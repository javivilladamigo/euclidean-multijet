'''
Imports and config
'''
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
import plots
import json
import itertools


# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #this doesn't work, need to run `conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1` and then reactivate the conda environment

np.random.seed(0)
torch.manual_seed(0)#make training results repeatable 
plots.update_rcparams()
SCREEN_WIDTH = 100

'''
Labels for classifier
'''
class ClassInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = ClassInfo(abbreviation='d4', name= 'FourTag Data', color='red')
d3 = ClassInfo(abbreviation='d3', name='ThreeTag Data', color='orange')

S = ClassInfo(abbreviation='S', name='Signal Data', color='red')
BG = ClassInfo(abbreviation='BG', name='Background Data', color='orange')

FvT_classes = [d4, d3]
SvB_classes = [BG, S]
for i, c in enumerate(FvT_classes): c.index = i
for i, c in enumerate(SvB_classes): c.index = i


'''
Load coffea files and convert to tensor
'''
def load(cfiles, selection=''):
    event_list = []
    for cfile in cfiles:
        print(cfile, selection)
        event = util.load(cfile)
        if selection:
            event = event[eval(selection)]
        event_list.append(event)
    return ak.concatenate(event_list)

# convert coffea objects in to pytorch tensors
train_valid_modulus = 3
def coffea_to_tensor(event, device='cpu', decode = False, kfold=False):
    j = torch.FloatTensor( event['Jet',('pt','eta','phi','mass')].to_numpy().view(np.float32).reshape(-1, 4, 4) ) # [event,jet,feature]
    j = j.transpose(1,2).contiguous() # [event,feature,jet]
    e = torch.LongTensor( np.asarray(event['event'], dtype=np.uint8) )%train_valid_modulus
    if kfold:
        return j, e
    w = torch.FloatTensor( event['weight'].to_numpy().view(np.float32) )
    R  = 1*torch.LongTensor( event['SB'].to_numpy().view(np.uint8) )
    R += 2*torch.LongTensor( event['SR'].to_numpy().view(np.uint8) )
    if device != 'cpu':
        j, w, R, e = j.to(device), w.to(device), R.to(device), e.to(device)

    if decode == False:
        y = torch.LongTensor( np.asarray(event['class'], dtype=np.uint8) )
        y = y.to(device)
        dataset = TensorDataset(j, y, w, R, e)
    else:
        y = None
        dataset = TensorDataset(j, w, R, e)
    return dataset


'''
Architecture hyperparameters
'''
bottleneck_dim = 6
permutations = list(itertools.permutations([0,1,2,3]))
loss_pt = False                 # whether to add pt to the loss of PxPyPzE
permute_input_jet = False       # whether to randomly permute the positions of input jets
rotate_phi = False              # whether to remove eta-phi invariances in the encoding
correct_DeltaR = False          # whether to correct DeltaR (in inference)

sample = 'fourTag_10x'

testing = False
plot_training_progress = True  # plot training progress
if testing:
    num_epochs = 1
    plot_every = 1
else:
    num_epochs = 25
    plot_every = 5

lr_init  = 0.01
lr_scale = 0.25
bs_scale = 2
bs_milestones =     [1, 3, 6, 10]
lr_milestones =     [10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 100, 200, 250, 300, 400, 450]
#gb_milestones =     [5, 7, 10, 15, 20, 30, 40, 50, 100, 150, 200]

train_batch_size = 2**10
infer_batch_size = 2**14
max_train_batch_size = train_batch_size*64
num_workers=8


############ default hyperparameters for FvT ############
# num_epochs = 20                                       #
# lr_init  = 0.01                                       #
# lr_scale = 0.25                                       #
# bs_scale = 2                                          #
#                                                       #
# bs_milestones = [1,3,6,10]                            #
# lr_milestones = [15,16,17,18,19,20,21,22,23,24]       #
#                                                       #
# train_batch_size = 2**10                              #
# infer_batch_size = 2**14                              #
# max_train_batch_size = train_batch_size*64            #
#                                                       #
# num_workers=8                                         #
#                                                       #
#########################################################

train_loss_tosave = [] # vectors to store loss during training for plotting afterwards (to be implemented)
val_loss_tosave = []



'''
Batch loaders class for inference and training
'''
class Loader_Result:
    def __init__(self, model, dataset, n_classes=2, train=False):
        self.dataset = dataset
        self.infer_loader = DataLoader(dataset=dataset, batch_size=infer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        self.train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True) if train else None
        self.n = len(dataset)
        self.w = dataset.tensors[1] if model.task == 'dec' else dataset.tensors[2]
        self.w_sum = self.w.sum()
        self.cross_entropy = torch.zeros(self.n)
        self.decoding_loss = torch.zeros(self.n)
        self.j_ = torch.zeros(self.n, 4, 4) # store vectors for plotting at the end of epoch   
        self.rec_j_ = torch.zeros(self.n, 4, 4)
        self.z_ = torch.zeros(self.n, model.network.d_bottleneck, 1) # activations in the embedded space
        self.m2j_ = torch.zeros(self.n, 1, 6)
        self.m4j_ = torch.zeros(self.n, 1, 3)
        self.rec_m2j_ = torch.zeros(self.n, 1, 6)
        self.rec_m4j_ = torch.zeros(self.n, 1, 3)
        self.component_weights = torch.tensor([1,1,0.3,0.3]).view(1,4,1) # adapt magnitude of PxPy versus PzE
        self.n_done = 0
        self.loaded_die_loss = model.loaded_die_loss if hasattr(model, 'loaded_die_loss') else None
        self.loss_estimate = 1.0
        self.history = {'loss': []}
        self.task = model.task
        self.train = train

    def eval(self):
        self.n_done = 0

    def loss_fn(self, jPxPyPzE, rec_jPxPyPzE, j, rec_j, phi_rotations, reduction = 'mean', ignore_perm = True):
        
        if ignore_perm: # do not search for the minimum loss of jets combination
            mse_loss_batch = F.mse_loss(jPxPyPzE*self.component_weights, rec_jPxPyPzE*self.component_weights, reduction = 'none').sum(dim = (1,2)) # sum along jets and features errors
        
        else:
            # compute all the possible 24 reconstruction losses between 0123 in output with 0123 in input
            jPxPyPzE = jPxPyPzE.unsqueeze(3).repeat(1, 1, 1, 24)                    # repeat jPxPyPzE (copy) along the 24-sized permutations dimension
            j = j.unsqueeze(3).repeat(1, 1, 1, 24)                                  # repeat j (copy) along the 24-sized permutations dimension
            
            
            rec_j_perm = torch.zeros(*rec_j.shape, 24)
            rec_jPxPyPzE_perm = torch.zeros(*rec_jPxPyPzE.shape, 24)
            for k, perm in enumerate(permutations):      # produce all possible jet permutations of reconstructed jets
                    rec_j_perm[:, :, :, k] = rec_j[:, :, perm]
                    rec_jPxPyPzE_perm[:, :, :, k] = rec_jPxPyPzE[:, :, perm]
            rec_j = rec_j_perm
            rec_jPxPyPzE = rec_jPxPyPzE_perm
        
            self.component_weights = self.component_weights.unsqueeze(2)                                                                                    # add a component for the permutations dimension in the case we are not ignoring perms
            mse_loss_batch_perms = F.mse_loss(jPxPyPzE*self.component_weights, rec_jPxPyPzE*self.component_weights, reduction = 'none').sum(dim = (1,2))    # sum along jets and features errors
            mse_loss_batch, perm_index = mse_loss_batch_perms.min(dim = 1)                                                                                  # dimension 0 is batch and dimension 1 is permutation
            rec_jPxPyPzE = rec_jPxPyPzE[torch.arange(rec_jPxPyPzE.shape[0]), :, :, perm_index]                                                              # re-obtain the [batch_number, 4, 4] tensor with the jet with minimum loss

            # loss on m2j (not employed)
            '''
            d, dPxPyPzE = networks.addFourVectors(0, 0, jPxPyPzE[:,:,(0,2,0,1,0,1),0], jPxPyPzE[:,:,(1,3,2,3,3,2),0])
            rec_d, rec_dPxPyPzE = networks.addFourVectors(0, 0, rec_jPxPyPzE[:,:,(0,2,0,1,0,1)], rec_jPxPyPzE[:,:,(1,3,2,3,3,2)])
            mass_loss = F.mse_loss(d[:, 3:4,:], rec_d[:, 3:4,:], reduction = 'none').sum(dim=(1,2))
            '''             
            # perm_index is a number 0-23 indicating the best permutation: 0123, 0132, 0213, ..., 3210
            



        # taking the sqrt so that [loss] = GeV
        loss_batch = (mse_loss_batch).sqrt() # loss_batch.shape = [batch_size]

        return loss_batch, rec_jPxPyPzE

    def infer_batch_AE(self, jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j, phi_rotations, epoch, plot_every): # expecting same sized j and rec_j
        n_batch = rec_jPxPyPzE.shape[0]
        loss_batch, rec_jPxPyPzE = self.loss_fn(jPxPyPzE, rec_jPxPyPzE, j, rec_j, phi_rotations = phi_rotations)

        if epoch % plot_every == 0 and self.train: # way of ensure we are saving jets and z from training dataset
            self.j_[self.n_done : self.n_done + n_batch] = jPxPyPzE
            self.rec_j_[self.n_done : self.n_done + n_batch] = rec_jPxPyPzE
            self.z_[self.n_done : self.n_done + n_batch] = z
            self.m2j_[self.n_done : self.n_done + n_batch] = m2j
            self.m4j_[self.n_done : self.n_done + n_batch] = m4j
            self.rec_m2j_[self.n_done : self.n_done + n_batch] = rec_m2j
            self.rec_m4j_[self.n_done : self.n_done + n_batch] = rec_m4j

        self.decoding_loss[self.n_done : self.n_done + n_batch] = loss_batch
        self.n_done += n_batch
    
    def infer_done_AE(self):
        self.loss = (self.w * self.decoding_loss).sum() / self.w_sum # 
        self.history['loss'].append(copy(self.loss))
        train_loss_tosave.append(self.loss.item()) if self.train else val_loss_tosave.append(self.loss.item()) # save loss to plot later
        self.n_done = 0

    def train_batch_AE(self, jPxPyPzE, rec_jPxPyPzE, j, rec_j, w, phi_rotations): # expecting same sized j and rec_j
        loss_batch, _ = self.loss_fn(jPxPyPzE, rec_jPxPyPzE, j, rec_j, phi_rotations=phi_rotations) # here rec_jPxPyPzE is not used; we only plot during inference of train dataset
        loss = (w * loss_batch).sum() / w.sum() # multiply weight for all the jet features and recover the original shape of the features 
        loss.backward()
        self.loss_estimate = self.loss_estimate * .02 + 0.98*loss.item()


'''
Model used for autoencoding
'''
class Model_AE:
    def __init__(self, train_valid_offset = 0, device = 'cpu', task = 'dec', model_file = '', sample = '', generate_synthetic_dataset = False):
        self.train_valid_offset = train_valid_offset
        self.device = device
        self.task = task
        self.sample = sample
        self.generate_synthetic_dataset = generate_synthetic_dataset
        self.return_masses = True # whether to return masses from the Input_Embed; this is used by the class member function K_fold
        self.network = networks.Basic_CNN_AE(dimension = 16, bottleneck_dim = bottleneck_dim, permute_input_jet = permute_input_jet, phi_rotations = rotate_phi, correct_DeltaR = correct_DeltaR, return_masses = self.return_masses, device = self.device) if not self.generate_synthetic_dataset else networks.Basic_decoder(dimension = 16, bottleneck_dim = bottleneck_dim, correct_DeltaR = correct_DeltaR, return_masses = self.return_masses, n_ghost_batches = 64, device = self.device)
        self.network.to(self.device)
        n_trainable_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f'Network has {n_trainable_parameters} trainable parameters')
        self.epoch = 0
        self.lr_current = lr_init
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr_init, amsgrad=False)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma=lr_scale, last_epoch=-1)
        self.lr_change = []
        self.bs_change = []
        self.n_done = 0

        if model_file:
            print(f'Load {model_file}')
            model_name_parts = re.split(r'[/_]', model_file)
            self.task = model_name_parts[1] if not self.generate_synthetic_dataset else "gen"
            print(f'The task of the model is {self.task}')
            self.train_valid_offset = model_name_parts[model_name_parts.index('offset')+1]
            self.model_pkl = model_file
            self.model_dict = torch.load(self.model_pkl)
            if self.generate_synthetic_dataset: # separate this as we do not need the optimizer
                self.network.load_state_dict(self.model_dict['decoder'])
                self.network.eval()
            else:
                self.network.load_state_dict(self.model_dict['model'])
                self.optimizer.load_state_dict(self.model_dict['optimizer'])
            self.epoch = self.model_dict['epoch']
        
    def make_loaders(self, event):
        '''
        Split into training and validation, define datasets and format into pytorch tensors, declare train_result and valid_result
        '''
        # Split into training and validation sets and format into pytorch tensors
        valid = event.event%train_valid_modulus == self.train_valid_offset
        train = ~valid

        dataset_train = coffea_to_tensor(event[train], device=self.device, decode = True)
        dataset_valid = coffea_to_tensor(event[valid], device=self.device, decode = True)

        self.train_result = Loader_Result(self, dataset_train, train=True)
        self.valid_result = Loader_Result(self, dataset_valid)

        print(f'{self.train_result.n:,} training samples split into {len(self.train_result.train_loader):,} batches of {train_batch_size:,}')
        print(f'{self.valid_result.n:,} validation samples split into {len(self.valid_result.infer_loader):,} batches of {infer_batch_size:,}')
    
    @torch.no_grad()
    def inference(self, result):
        '''
        Reconstruct the jets in inference mode and compute the loss
        '''
        self.network.eval()

        # nb, event jets vector, weight, region, event number
        for batch_number, (j, w, R, e) in enumerate(result.infer_loader):
            jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j = self.network(j)

            
            result.infer_batch_AE(jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j, self.network.phi_rotations, epoch = self.epoch, plot_every = plot_every)
            
            percent = float(batch_number+1)*100/len(result.infer_loader)
            sys.stdout.write(f'\rEvaluating {percent:3.0f}%')
            sys.stdout.flush()

        result.infer_done_AE()

    def train_inference(self):
        self.inference(self.train_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'\n\nEpoch {self.epoch:>2} | Training   | Loss {self.train_result.loss:1.5} GeV')

    def valid_inference(self):
        self.inference(self.valid_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'         | Validation | Loss {self.valid_result.loss:1.5} GeV')

    def train(self, result=None):
        if result is None: result = self.train_result
        self.network.train() # inherited from nn.Module()

        for batch_number, (j, w, R, e) in enumerate(result.train_loader):
            self.optimizer.zero_grad()

            jPxPyPzE, rec_jPxPyPzE, j, rec_j, z, m2j, m4j, rec_m2j, rec_m4j = self.network(j)

            result.train_batch_AE(jPxPyPzE, rec_jPxPyPzE, j, rec_j, w, self.network.phi_rotations)

            percent = float(batch_number+1)*100/len(result.train_loader)
            sys.stdout.write(f'\rTraining {percent:3.0f}% >>> Loss Estimate {result.loss_estimate:1.5f} GeV')
            sys.stdout.flush()
            self.optimizer.step()
        result.loss_estimate = 0

    def increment_train_loader(self, new_batch_size = None):
        current_batch_size = self.train_result.train_loader.batch_size
        if new_batch_size is None: new_batch_size = current_batch_size * bs_scale
        if new_batch_size == current_batch_size: return
        print(f'Change training batch size: {current_batch_size} -> {new_batch_size} ({self.train_result.n//new_batch_size} batches)')
        del self.train_result.train_loader
        self.train_result.train_loader = DataLoader(dataset=self.train_result.dataset, batch_size=new_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        self.bs_change.append(self.epoch)

    def run_epoch(self, plot_training_progress):
        
        self.epoch += 1
        self.train()
        self.train_inference()        
        self.valid_inference()
        self.scheduler.step()
        if plot_training_progress and self.epoch % plot_every == 0:
            plots.plot_training_residuals_PxPyPzEm2jm4jPtm2jvsm4j(jPxPyPzE=self.train_result.j_, rec_jPxPyPzE=self.train_result.rec_j_, phi_rot=self.network.phi_rotations, m2j=self.train_result.m2j_, m4j=self.train_result.m4j_, rec_m2j=self.train_result.rec_m2j_, rec_m4j=self.train_result.rec_m4j_, offset=self.train_valid_offset, epoch=self.epoch, sample=self.sample, network_name=self.network.name) # plot training residuals for pt, eta, phi
            plots.plot_PxPyPzEPtm2jm4j(jPxPyPzE=self.train_result.j_, rec_jPxPyPzE=self.train_result.rec_j_, phi_rot=self.network.phi_rotations, m2j=self.train_result.m2j_, m4j=self.train_result.m4j_, rec_m2j=self.train_result.rec_m2j_, rec_m4j=self.train_result.rec_m4j_, offset=self.train_valid_offset, epoch=self.epoch, sample=self.sample, network_name=self.network.name)
            randomly_plotted_event_number = plots.plot_etaPhi_plane(jPxPyPzE = self.train_result.j_, rec_jPxPyPzE = self.train_result.rec_j_, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)
            plots.plot_PxPy_plane(jPxPyPzE = self.train_result.j_, rec_jPxPyPzE = self.train_result.rec_j_, event_number = randomly_plotted_event_number, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)
            plots.plot_activations_embedded_space(z=self.train_result.z_, offset=self.train_valid_offset, epoch=self.epoch, sample=self.sample, network_name=self.network.name)


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

    def run_training(self, plot_training_progress = False):
        min_val_loss = 1e20
        val_loss_increase_counter = 0
        self.network.set_mean_std(self.train_result.dataset.tensors[0])
        # self.train_inference()
        # self.valid_inference()


        
        for _ in range(num_epochs):
            self.run_epoch(plot_training_progress = plot_training_progress)
            
            if val_loss_tosave[-1] < min_val_loss and _ > 0:
                self.del_prev_model()
                self.save_model()
                min_val_loss = val_loss_tosave[-1]
                val_loss_increase_counter = 0
            else:
                val_loss_increase_counter += 1
            
            if val_loss_increase_counter == 20:
                val_loss_increase_counter = 0
                print(f'Val loss has not decreased in 20 epoch. Decay learning rate: {self.lr_current} -> {self.lr_current*lr_scale}')
                self.lr_current *= lr_scale
                self.lr_change.append(self.epoch)
            #if val_loss_increase_counter == 100: #or min_val_loss < 1.:
                #break
            
        loss_tosave = {"train" : train_loss_tosave, "val" : val_loss_tosave}
        with open("loss.txt", 'w') as file:
            file.write(json.dumps(loss_tosave))
        plots.plot_loss(loss_tosave, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)        
    
    def save_model(self):
        self.history = {'train': self.train_result.history,
                        'valid': self.valid_result.history}
        self.model_dict = {'model': deepcopy(self.network.state_dict()),
                           'decoder': deepcopy(self.network.decoder.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict()),
                           'epoch': self.epoch,
                           'history': copy(self.history)}
        self.model_pkl = f'models/{self.task}_{self.sample}_{self.network.name}_offset_{self.train_valid_offset}_epoch_{self.epoch:03d}.pkl'
        print(f'Saved model as: {self.model_pkl} with a validation loss {val_loss_tosave[-1]:.2e}')
        torch.save(self.model_dict, self.model_pkl)
    
    def del_prev_model(self):
        self.prev_models = glob(f'models/{self.task}_{self.sample}_{self.network.name}_offset_{self.train_valid_offset}_epoch_*.pkl')
        for model in self.prev_models:
            os.remove(model)




'''
Arguments for execution
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default=False, action='store_true', help='Run model training')
    parser.add_argument('-tk', '--task', default='FvT', type = str, help='Type of classifier (FvT or SvB) to run')
    parser.add_argument('-o', '--offset', default=0, type=int, help='k-folding offset for split between training/validation sets')
    parser.add_argument('-m', '--model', default='', type=str, help='Load these models (* wildcard for offsets)')
    parser.add_argument('-g', '--generate', default=False, action='store_true', help='To be passed with --model and the specific models to generate data')
    args = parser.parse_args()

    
    custom_selection = 'event.preselection' # region on which you want to train



    '''
    Training
    '''
    if args.train:
        if args.task:
            task = args.task
        else:
            # demand the specification of --task if --train to avoid conflicts
            sys.exit("Task not specified. Use FvT, SvB or dec. Exiting...")

        classes = FvT_classes if task == 'FvT' else SvB_classes if task == 'SvB' else None

        # task is fourTag vs. threeTag classification
        if task == 'FvT':
            coffea_4b = sorted(glob('data/fourTag_picoAOD*.coffea'))
            coffea_3b = sorted(glob('data/threeTag_picoAOD*.coffea'))

            # Load data
            event_3b = load(coffea_3b, selection=custom_selection)
            event_4b = load(coffea_4b, selection=custom_selection)

            # Generate labels
            event_3b['d3'] = True
            event_3b['d4'] = False
            event_4b['d3'] = False
            event_4b['d4'] = True

            # Form events with threeTag + fourTag
            event = ak.concatenate([event_3b, event_4b])

            # Assign labels to each particular event
            event['class'] = d4.index*event.d4 + d3.index*event.d3 # for binary classification this seems dumb, makes sense when you have multi-class classification

            # Load model and run training
            model_args = {  'task': task,
                            'train_valid_offset': args.offset}
            t=Model(**model_args) # Four vs Threetag classification
            t.make_loaders(event)
            t.run_training()



        # task is Signal vs. Background classification
        if task == 'SvB': 
            coffea_signal = sorted(glob('data/HH4b_picoAOD*.coffea'))
            coffea_background = sorted(glob('data/threeTag_picoAOD*.coffea')) # file used for background

            # Load data
            event_signal = load(coffea_signal, selection=custom_selection)
            event_background = load(coffea_background, selection=custom_selection)

            # Generate labels
            event_signal['S'] = True
            event_signal['BG'] = False
            event_background['S'] = False
            event_background['BG'] = True

            # Form events with signal + background
            event = ak.concatenate([event_signal, event_background])

            # Assign labels to each particular event
            event['class'] = S.index*event.S + BG.index*event.BG # for binary classification this seems dumb, makes sense when you have multi-class classification

            # Load model and run training
            model_args = {  'task': task,
                            'train_valid_offset': args.offset}
            t=Model(**model_args)
            t.make_loaders(event)
            t.run_training()


        # task is autoencoding
        if task == 'dec':
            coffea_file = sorted(glob(f'data/{sample}_picoAOD*.coffea')) # file used for autoencoding
            
            # Load data
            event = load(coffea_file, selection = custom_selection)

            # Load model and run training
            model_args = {  'task': task,
                            'train_valid_offset': args.offset}
            t=Model_AE(**model_args, sample = sample)
            t.make_loaders(event)
            t.run_training(plot_training_progress = plot_training_progress)

            
            
    
    
    
    '''
    Pre-compute friend TTrees with the validation results after training
    '''
    if args.model and not args.generate:
        # task is specified as the three letters before "_Basic" in model filename
        task = args.model[0:3] if '/' not in args.model else args.model[args.model.find('/') + 1 : args.model.find('/') + 4]
        # Task is classification (either FvT or SvB)
        if task == 'FvT' or task == 'SvB':
            classes = FvT_classes if task == 'FvT' else SvB_classes if task == 'SvB' else None

            model_files = sorted(glob(args.model))
            models = []
            for model_file in model_files:
                models.append(Model(model_file=model_file))

            task = models[0].task
            kfold = networks.K_Fold([model.network for model in models], task = task)

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
                    
                    # If I want to keep for now the original branches of the coffea files
                    '''for branch in event.fields:
                        kfold_dict[f'{branch}'] = event.__getattr__(branch)'''
                    # the upper loop gives problems so I'll just stick to the most relevant ones
                    kfold_dict["preselection"] = np.array(event.preselection)
                    kfold_dict["SR"] = np.array(event.SR)
                    kfold_dict["SB"] = np.array(event.SB)


                    for cl in classes:
                        kfold_dict[cl.abbreviation] = c_score[:,cl.index].numpy()
                    
                    if task == 'FvT':
                        kfold_dict['rw'] = kfold_dict['d4'] / kfold_dict['d3']                
                    if task == 'SvB':
                        kfold_dict['ratio_SvB'] = kfold_dict['S'] / kfold_dict['BG']
                    
                    output['Events'] = {task: ak.zip(kfold_dict),
                                        'event': event.event}

        # task is autoencoding
        elif task == 'dec':
            model_files = sorted(glob(args.model))
            models = []
            for model_file in model_files:
                models.append(Model_AE(model_file=model_file))

            d = models[0].network.d_bottleneck
            epoch_string = model_files[0][model_files[0].find('epoch') + 6 : model_files[0].find('epoch')+ 9]

            kfold = networks.K_Fold([model.network for model in models], task = task)

            import uproot
            import awkward as ak

            picoAODs = glob('data/*picoAOD.root')
            for picoAOD in picoAODs:
                output_file = picoAOD.replace('picoAOD', task)
                print(f'Generate kfold output for {picoAOD} -> {output_file}')
                coffea_files = sorted(glob(picoAOD.replace('.root','*.coffea')))
                event = load(coffea_files)
                j, e = coffea_to_tensor(event, decode = True, kfold=True)
                rec_j, z = kfold(j, e) # output reconstructed jets and embedded space

                activation_file = picoAOD.replace('data/', 'activations/').replace('picoAOD.root', f'z_{d}_epoch_{epoch_string}.pkl')
                plots.mkpath("activations/")
                torch.save({'activations' : z}, activation_file)
                print(f"Saved embedded space tensor to {activation_file}")

                # create np arrays to fill each element with the 4 quantities corresponding to the event
                pt_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))
                eta_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))
                phi_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))
                mass_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))

                with uproot.recreate(output_file) as output:
                    kfold_dict = {}

                    for jet_nb in range(rec_j.shape[2]): # go through each of the 4 jets
                        pt_array[:, jet_nb]     = rec_j[:, 0, jet_nb].numpy()
                        eta_array[:, jet_nb]    = rec_j[:, 1, jet_nb].numpy()
                        phi_array[:, jet_nb]    = rec_j[:, 2, jet_nb].numpy()
                        mass_array[:, jet_nb]   = rec_j[:, 3, jet_nb].numpy()

                    # Store jet properties as 2D arrays in kfold_dict
                    kfold_dict['Jet_pt'] = pt_array
                    kfold_dict['Jet_eta'] = eta_array
                    kfold_dict['Jet_phi'] = phi_array
                    kfold_dict['Jet_mass'] = mass_array

                    # Write the dict to the output file
                    output["Events"] = kfold_dict

        else:
            sys.exit("Task not found in model filename. Write models/(dec, SvB, FvT)_Basic[...]")
    
    '''
    Pre-compute friend TTrees with synthetic datasets
    '''
    if args.generate:
        task = 'gen'
        model_files = sorted(glob(args.model))
        models = []
        for model_file in model_files:
            models.append(Model_AE(model_file=model_file, task = task, generate_synthetic_dataset = True))
        
        d = models[0].network.d_bottleneck
        epoch_string = model_files[0][model_files[0].find('epoch') + 6 : model_files[0].find('epoch')+ 9]

        kfold = networks.K_Fold([model.network for model in models], task = task)

        import uproot
        import awkward as ak

        plots.mkpath("activations/")
        picoAODs = glob('data/*picoAOD.root')
        for picoAOD in picoAODs:
            activations_file = picoAOD.replace('data/', 'activations/').replace('picoAOD.root', f'z_{d}_epoch_{epoch_string}.pkl')
            output_file = picoAOD.replace('picoAOD', task)
            print(f'Generate kfold generated output for {activations_file} -> {output_file}')
            activations = torch.load(activations_file)["activations"]
            e = torch.LongTensor(np.arange(activations.shape[0], dtype=np.uint8)) % train_valid_modulus
            print("loaded z shape from picoAOD:", activations.shape)
            rec_j = kfold(activations, e)
            print("extracted kfold jets shape:", rec_j.shape)

            # create np arrays to fill each element with the 4 quantities corresponding to the event
            pt_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))
            eta_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))
            phi_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))
            mass_array = np.zeros((rec_j.shape[0], rec_j.shape[2]))

            with uproot.recreate(output_file) as output:
                kfold_dict = {}

                for jet_nb in range(rec_j.shape[2]): # go through each of the 4 jets
                    pt_array[:, jet_nb]     = rec_j[:, 0, jet_nb].numpy()
                    eta_array[:, jet_nb]    = rec_j[:, 1, jet_nb].numpy()
                    phi_array[:, jet_nb]    = rec_j[:, 2, jet_nb].numpy()
                    mass_array[:, jet_nb]   = rec_j[:, 3, jet_nb].numpy()

                # Store jet properties as 2D arrays in kfold_dict
                kfold_dict['Jet_pt'] = pt_array
                kfold_dict['Jet_eta'] = eta_array
                kfold_dict['Jet_phi'] = phi_array
                kfold_dict['Jet_mass'] = mass_array

                # Write the dict to the output file
                output["Events"] = kfold_dict
    
    if not args.train and not args.model and not args.generate:
        sys.exit("No --train nor --model specified. Script is not training nor precomputing friend TTrees. Exiting...")
