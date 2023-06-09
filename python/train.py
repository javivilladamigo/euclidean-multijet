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
from torch.utils.tensorboard import SummaryWriter

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #this doesn't work, need to run `conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1` and then reactivate the conda environment

np.random.seed(0)
torch.manual_seed(0)#make training results repeatable 

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
testing = True
if testing:
    num_epochs = 20
    plot_every = 1
else:
    num_epochs = 500
    plot_every = 50

lr_init  = 0.01
lr_scale = 0.5
bs_scale = 2

bs_milestones =     [50, 100, 200, 275]
lr_milestones =     [50, 100, 200, 250, 300, 400, 450]
gb_milestones =     [5, 7, 10, 15, 20, 30, 40, 50, 100, 150, 200]

train_batch_size = 2**9
infer_batch_size = 2**13
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
min_val_loss = 999



'''
Batch loaders class for inference and training
'''
class Loader_Result:
    def __init__(self, model, dataset, n_classes=2, train=False):
        self.dataset = dataset
        self.infer_loader = DataLoader(dataset=dataset, batch_size=infer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        self.train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=False,  num_workers=num_workers, pin_memory=True, drop_last=True) if train else None
        self.n = len(dataset)
        self.w = dataset.tensors[1] if model.task == 'dec' else dataset.tensors[2]
        self.w_sum = self.w.sum()
        self.cross_entropy = torch.zeros(self.n)


        self.decoding_loss = torch.zeros(self.n)  # [batch_size, nb_of_features, effective_nb_of_jets]
                                                        # nb_of_features is the number of features reconstructed (i.e. how many features from [pt, eta, phi, mass])
                                                        # effective_nb_of_jets is the multiplicity of values reconstructed for each feature: 3 if only 3 relative features are being reco'd, leaving one degree of freedom, 4 if all relative pairings (i.e. 12, 23, 34, 14) are # # # to be reconstructed. Keep 3 for reconstructing only 12, 23, 34 pairings.
        self.n_done = 0
        self.loaded_die_loss = model.loaded_die_loss if hasattr(model, 'loaded_die_loss') else None
        self.loss_estimate = 1.0
        self.history = {'loss': []}
        self.task = model.task
        self.train = train

    def eval(self):
        self.n_done = 0

    def loss_fn(self, j, rec_j):

        mse_loss_batch_Px = F.mse_loss(j[:, 0:1, :], rec_j[:, 0:1, :], reduction = 'none') # compute the MSE loss between reconstructed jets and input jets
        mse_loss_batch_Py = F.mse_loss(j[:, 1:2, :], rec_j[:, 1:2, :], reduction = 'none') # don't use Py of leading jet because it is always 0 when phi_lead = 0
        mse_loss_batch_Pz = F.mse_loss(j[:, 2:3, :], rec_j[:, 2:3, :], reduction = 'none') # compute the MSE loss between reconstructed jets and input jets
        mse_loss_batch_E = F.mse_loss(j[:, 3:4, :],  rec_j[:, 3:4, :],  reduction = 'none') # compute the MSE loss between reconstructed jets and input jets
        mse_loss_batch = torch.cat([mse_loss_batch_Px, mse_loss_batch_Py, mse_loss_batch_Pz, mse_loss_batch_E], dim = 1)
        
        mse_loss_batch = mse_loss_batch.sum(dim=(1,2))

        return mse_loss_batch

    def infer_batch_AE(self, j, rec_j): # expecting same sized j and rec_j
        n_batch = rec_j.shape[0]

        loss_batch = self.loss_fn(j, rec_j)

        self.decoding_loss[self.n_done : self.n_done + n_batch] = loss_batch
        self.n_done += n_batch
    
    def infer_done_AE(self):
        print("\nMean infer loss:", self.decoding_loss.mean(dim=0).data)
        self.loss = (self.w * self.decoding_loss).sum() / self.w_sum # multiply weight for all the jet features and recover the original shape of the features 
        self.history['loss'].append(copy(self.loss))
        train_loss_tosave.append(self.loss.item()) if self.train else val_loss_tosave.append(self.loss.item())
        self.n_done = 0

    def train_batch_AE(self, j, rec_j, w): # expecting same sized j and rec_j

        loss_batch = self.loss_fn(j, rec_j)

        loss_batch = (w * loss_batch).sum() / w.sum() # multiply weight for all the jet features and recover the original shape of the features 
        loss_batch.backward()
        self.loss_estimate = self.loss_estimate * .02 + 0.98*loss_batch.item()


'''
Model used for autoencoding
'''
class Model_AE:
    def __init__(self, train_valid_offset = 0, device = 'cpu', task = 'dec', model_file = '', sample = ''):
        self.task = task
        self.device = device
        self.train_valid_offset = train_valid_offset
        self.sample = sample
        self.network = networks.Basic_CNN_AE(dimension = 8, device = self.device)
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
            rec_jPxPyPzE, rec_jPxPyPzE_sc, jPxPyPzE, jPxPyPzE_sc, rec_m2j, m2j, rec_m4j, m4j = self.network(j)

            
            result.infer_batch_AE(jPxPyPzE_sc[:,0:4,:], rec_jPxPyPzE_sc[:,0:4,:])
            
            percent = float(batch_number+1)*100/len(result.infer_loader)
            sys.stdout.write(f'\rEvaluating {percent:3.0f}%')
            sys.stdout.flush()

        result.infer_done_AE()


    def train_inference(self):
        self.inference(self.train_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'\n\nEpoch {self.epoch:>2} | Training | Loss {self.train_result.loss:1.5}')

    def valid_inference(self):
        self.inference(self.valid_result)
        sys.stdout.write(' '*SCREEN_WIDTH)
        sys.stdout.flush()
        print('\r',end='')
        print(f'         | Validation | Loss {self.valid_result.loss:1.5}')

    def train(self, result=None):
        if result is None: result = self.train_result
        self.network.train() # inherited from nn.Module()

        for batch_number, (j, w, R, e) in enumerate(result.train_loader):
            self.optimizer.zero_grad()

            rec_jPxPyPzE, rec_jPxPyPzE_sc, jPxPyPzE, jPxPyPzE_sc, rec_m2j, m2j, rec_m4j, m4j = self.network(j)

            
            result.train_batch_AE(jPxPyPzE_sc[:,0:4,:], rec_jPxPyPzE_sc[:,0:4,:], w)

            percent = float(batch_number+1)*100/len(result.train_loader)
            sys.stdout.write(f'\rTraining {percent:3.0f}% >>> Loss Estimate {result.loss_estimate:1.5f}')
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

    def run_epoch(self, plot_res):
        

        self.epoch += 1
        self.train()
        self.train_inference()        
        self.valid_inference()
        self.scheduler.step()

        if plot_res and self.epoch % plot_every == 0:
            total_jPxPyPzE_ = torch.Tensor(())
            total_rec_jPxPyPzE_ = torch.Tensor(())
            total_m2j_ = torch.Tensor(())
            total_rec_m2j_ = torch.Tensor(())
            total_m4j_ = torch.Tensor(())
            total_rec_m4j_ = torch.Tensor(())



            for i, (j_, w_, R_, e_) in enumerate(self.train_result.infer_loader):
                rec_jPxPyPzE_, rec_jPxPyPzE_sc_, jPxPyPzE_, jPxPyPzE_sc_, rec_m2j_, m2j_, rec_m4j_, m4j_ = self.network(j_) # forward pass

                total_m2j_ = torch.cat((total_m2j_, m2j_), 0)
                total_rec_m2j_ = torch.cat((total_rec_m2j_, rec_m2j_), 0)

                total_m4j_ = torch.cat((total_m4j_, m4j_), 0)
                total_rec_m4j_ = torch.cat((total_rec_m4j_, rec_m4j_), 0)


                total_j_ = torch.cat((total_jPxPyPzE_, jPxPyPzE_), 0)
                total_rec_j_ = torch.cat((total_rec_jPxPyPzE_, rec_jPxPyPzE_), 0)





            plots.plot_training_residuals_PxPyPzEm2jm4jPt(total_j_[:,0:4,:], total_rec_j_[:,0:4,:], total_m2j_, total_rec_m2j_, total_m4j_, total_rec_m4j_, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name) # plot training residuals for pt, eta, phi
            plots.plot_PxPyPzE(total_j_[:,0:4,:], total_rec_j_[:,0:4,:], offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)



        if (self.epoch in bs_milestones or self.epoch in lr_milestones or self.epoch in gb_milestones) and self.network.n_ghost_batches:
            if self.epoch in gb_milestones and self.network.n_ghost_batches:
                gb_decay = 4 # 2 if self.epoch in bs_mile
                print(f'set_ghost_batches({self.network.n_ghost_batches//gb_decay})')
                self.network.set_ghost_batches(self.network.n_ghost_batches//gb_decay)
            if self.epoch in bs_milestones:
                self.increment_train_loader()
            if self.epoch in lr_milestones:
                print(f'Decay learning rate: {self.lr_current} -> {self.lr_current*lr_scale}')
                self.lr_current *= lr_scale
                self.lr_change.append(self.epoch)

    def run_training(self, plot_res = False):
        min_val_loss = 999
        val_loss_increase_counter = 0
        #self.network.set_mean_std(self.train_result.dataset.tensors[0])
        self.train_inference()
        self.valid_inference()

        tb = SummaryWriter()
        
        for _ in range(num_epochs):
            self.run_epoch(plot_res = plot_res)
            '''
            for name, weight in self.network.named_parameters():
                
                tb.add_histogram(name, weight, self.epoch)
                tb.add_histogram(f'{name}.grad',weight.grad, self.epoch)
            '''
            
            tb.add_scalar("Train loss", train_loss_tosave[-1], self.epoch)
            tb.add_scalar("Val loss", val_loss_tosave[-1], self.epoch)
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

        tb.flush()
        tb.close()
            
        loss = {"train" : train_loss_tosave, "val" : val_loss_tosave}
        with open("loss.txt", 'w') as file:
            file.write(json.dumps(loss))
        plots.plot_loss(loss, offset = self.train_valid_offset, epoch = self.epoch, sample = self.sample, network_name = self.network.name)
        
    
    def save_model(self):
        self.history = {'train': self.train_result.history,
                        'valid': self.valid_result.history}
        self.model_dict = {'model': deepcopy(self.network.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict()),
                           'epoch': self.epoch,
                           'history': copy(self.history)}
        self.model_pkl = f'models/{self.task}_{self.network.name}_offset_{self.train_valid_offset}_epoch_{self.epoch:03d}.pkl'
        print(f'Saved model as: {self.model_pkl} with a validation loss {val_loss_tosave[-1]:.2f}')
        torch.save(self.model_dict, self.model_pkl)
    
    def del_prev_model(self):
        self.prev_models = glob(f'models/{self.task}_{self.network.name}_offset_{self.train_valid_offset}_epoch_*.pkl')
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
            sample = 'fourTag'
            coffea_file = sorted(glob(f'data/{sample}_picoAOD*.coffea')) # file used for autoencoding
            
            # Load data
            event = load(coffea_file, selection = custom_selection)

            # Load model and run training
            model_args = {  'task': task,
                            'train_valid_offset': args.offset}
            t=Model_AE(**model_args, sample = sample)
            t.make_loaders(event)
            t.run_training(plot_res = True)

            
            
    
    
    
    '''
    Pre-compute friend TTrees with the validation results after training
    '''
    if args.model:
        
        # task is specified as the three letters before "_Basic" in model filename
        task = args.model[args.model.find('_Basic') - 3 : args.model.find('_Basic')]
        
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
                j, e = coffea_to_tensor(event, decode = True, kfold=True)
                rec_j = kfold(j, e)
                rec_j = networks.PtEtaPhiM(rec_j)
                print(rec_j.shape)


                with uproot.recreate(output_file) as output:
                    kfold_dict = {}


                    kfold_dict['nJet'] = rec_j.shape[2]
                    for jet_nb in range(1, rec_j.shape[2] + 1):
                        
                        kfold_dict[f'Jet{jet_nb}_pt']   = rec_j[:, 0, jet_nb - 1].numpy()
                        kfold_dict[f'Jet{jet_nb}_eta']  = rec_j[:, 1, jet_nb - 1].numpy()
                        kfold_dict[f'Jet{jet_nb}_phi']  = rec_j[:, 2, jet_nb - 1].numpy()
                        kfold_dict[f'Jet{jet_nb}_mass'] = rec_j[:, 3, jet_nb - 1].numpy()

                    #kfold_dict["preselection"] = np.array(event.preselection)
                    #kfold_dict["SR"] = np.array(event.SR)
                    #kfold_dict["SB"] = np.array(event.SB)
                    
                    output['Events'] = {task: ak.zip(kfold_dict)}
        else:
            sys.exit("Task not found in model filename. Write models/(dec, SvB, FvT)_Basic[...]")
    
    if not args.train and not args.model:
        sys.exit("No --train nor --model specified. Script is not training nor precomputing friend TTrees. Exiting...")