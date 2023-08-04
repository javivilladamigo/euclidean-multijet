import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import plots
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import norm

torch.manual_seed(0)#make training results repeatable 

def vector_print(vector, end='\n'):
    vectorString = ", ".join([f'{element:7.2f}' for element in vector])
    print(vectorString, end=end)



class Ghost_Batch_Norm(nn.Module): #https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition. 
    def __init__(self, features, ghost_batch_size=32, number_of_ghost_batches=64, n_averaging=1, stride=1, eta=0.9, bias=True, device='cpu', name='', conv=False, conv_transpose=False, features_out=None): # number_of_ghost_batches was initially set to 64
        super(Ghost_Batch_Norm, self).__init__()
        self.name = name
        self.index = None
        self.stride = stride if not conv_transpose else 1
        self.device = device
        self.features = features
        self.features_out = features_out if features_out is not None else self.features
        self.register_buffer('ghost_batch_size', torch.tensor(ghost_batch_size, dtype=torch.long))
        self.register_buffer('n_ghost_batches', torch.tensor(number_of_ghost_batches*n_averaging, dtype=torch.long))
        self.conv = False
        self.gamma = None
        self.bias = None
        self.updates = 0
        if conv:
            self.conv = nn.Conv1d(self.features, self.features_out, stride, stride=stride, bias=bias)
        elif conv_transpose:
            self.conv = nn.ConvTranspose1d(self.features, self.features_out, stride, stride=stride, bias=bias)
        else:
            self.gamma = nn.Parameter(torch .ones(self.features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.features))
        self.running_stats = True
        self.initialized   = False

        self.register_buffer('eps',  torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer('eta',  torch.tensor(eta, dtype=torch.float))
        self.register_buffer('m',    torch.zeros((1,1,self.stride,self.features), dtype=torch.float))
        self.register_buffer('s',    torch.ones ((1,1,self.stride,self.features), dtype=torch.float))
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float))
        self.register_buffer('one',  torch.tensor(1., dtype=torch.float))
        self.register_buffer('two',  torch.tensor(2., dtype=torch.float))

    def print(self):
        print('-'*50)
        print(self.name)
        for i in range(self.stride):
            print(" mean ",end='')
            vector_print(self.m[0,0,i,:])
        for i in range(self.stride):
            print("  std ", end='')
            vector_print(self.s[0,0,i,:])
        if self.gamma is not None:
            print("gamma ", end='')
            vector_print(self.gamma.data)
            if self.bias is not None:
                print(" bias ", end='')
                vector_print(self.bias.data)
        print()
        
    @torch.no_grad()
    def set_mean_std(self, x):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        pixel_groups = pixels//self.stride
        x = x.detach().transpose(1,2).contiguous().view(batch_size*pixels, 1, self.features)
        # this won't work for any layers with stride!=1
        x = x.view(-1, 1, self.stride, self.features)            
        m64 = x.mean(dim=0, keepdim=True, dtype=torch.float64)#.to(self.device)
        self.m = m64.type(torch.float32).to(self.device)
        self.s = x .std(dim=0, keepdim=True).to(self.device)
        self.initialized = True
        self.running_stats = False
        self.print()        

    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = torch.tensor(n_ghost_batches, dtype=torch.long).to(self.device)

    def forward(self, x, debug=False):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        pixel_groups = pixels//self.stride

        if self.training and self.n_ghost_batches!=0:
            # this has been changed from self.ghost_batch_size = batch_size // self.n_ghost_batches.abs()
            self.ghost_batch_size = torch.div(batch_size, self.n_ghost_batches.abs(), rounding_mode = 'trunc')

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = x.transpose(1,2).contiguous().view(self.n_ghost_batches.abs(), self.ghost_batch_size*pixel_groups, self.stride, self.features)
            
            gbm =  x.mean(dim=1, keepdim=True)
            gbs = (x. var(dim=1, keepdim=True) + self.eps).sqrt()

            #
            # Keep track of running mean and standard deviation. 
            #
            if self.running_stats or debug:
                # Use mean over ghost batches for running mean and std
                bm = gbm.detach().mean(dim=0, keepdim=True)
                bs = gbs.detach().mean(dim=0, keepdim=True)

                if debug and self.initialized:
                    gbms = gbm.detach().std(dim=0, keepdim=True)
                    gbss = gbs.detach().std(dim=0, keepdim=True)
                    m_pulls = (bm-self.m)/gbms
                    s_pulls = (bs-self.s)/gbss
                    #s_ratio = bs/self.s
                    #if (m_pulls.abs()>5).any() or (s_pulls.abs()>5).any():
                    print()
                    print(self.name)
                    print('self.m\n',self.m)
                    print('    bm\n', bm)
                    print('  gbms\n',gbms)
                    print('m_pulls\n',m_pulls,m_pulls.abs().mean(),m_pulls.abs().max())
                    print('-------------------------')
                    print('self.s\n',self.s)
                    print('    bs\n',bs)
                    print('  gbss\n',gbss)
                    print('s_pulls\n',s_pulls,s_pulls.abs().mean(),s_pulls.abs().max())
                    #print('s_ratio\n',s_ratio)
                    print()
                    #input()
                    
            if self.running_stats:
                # Simplest possible method
                if self.initialized:
                    self.m = self.eta*self.m + (self.one-self.eta)*bm
                    self.s = self.eta*self.s + (self.one-self.eta)*bs
                else:
                    self.m = self.zero*self.m+bm
                    self.s = self.zero*self.s+bs
                    self.initialized = True

            if self.n_ghost_batches>0:
                x = x - gbm
                x = x / gbs
            else:
                x = x.view(batch_size, pixel_groups, self.stride, self.features)
                x = x - self.m
                x = x / self.s
                
        else:
            # Use mean and standard deviation buffers rather than batch statistics
            #.view(self.n_ghost_batches, self.ghost_batch_size*pixel_groups, self.stride, self.features)
            x = x.transpose(1,2).view(batch_size, pixel_groups, self.stride, self.features)
            x = x - self.m
            x = x / self.s

        if self.conv:
            # back to standard indexing for convolutions: [batch, feature, pixel]
            x = x.view(batch_size, pixels, self.features).transpose(1,2).contiguous()
            x = self.conv(x)
        else:
            x = x * self.gamma
            if self.bias is not None:
                x = x + self.bias
            # back to standard indexing for convolutions: [batch, feature, pixel]
            x = x.view(batch_size, pixels, self.features).transpose(1,2).contiguous()
        return x            



#
# some basic four-vector operations
#
def PxPyPzE(v): # need this to be able to add four-vectors
    pt  = v[:,0:1]
    eta = v[:,1:2]
    phi = v[:,2:3]
    m   = v[:,3:4]
    
    Px, Py, Pz = pt*phi.cos(), pt*phi.sin(), pt*eta.sinh()
    E = (pt**2 + Pz**2 + m**2).sqrt()

    return torch.cat( (Px,Py,Pz,E), 1 )

def PtEtaPhiM(v):
    px = v[:,0:1]
    py = v[:,1:2]
    pz = v[:,2:3]
    e  = v[:,3:4]

    Pt  = (px**2+py**2).sqrt()
    ysign = 1-2*(py<0).float() # if py==0, px==Pt and acos(1)=pi/2 so we need zero protection on py.sign() --> changed to the current shape to avoid 0-gradient of .sign()
    Phi = (px/Pt).acos() * ysign
    Eta = (pz/Pt).asinh()

    M = F.relu(e**2 - px**2 - py**2 - pz**2).sqrt()

    return torch.cat( (Pt, Eta, Phi, M) , 1 ) 
    
def addFourVectors(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None): # output added four-vectors
    #vX[batch index, (pt,eta,phi,m), object index]
    
    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)

    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    v12        = PtEtaPhiM(v12PxPyPzE)


    return v12, v12PxPyPzE

def calcDeltaEta(v1, v2): # expects PtEtaPhiM representation
    dEta = (v2[:,1:2,:]-v1[:,1:2,:])
    return dEta
    
def calcDeltaPhi(v1, v2): #expects eta, phi representation
    dPhi12 = (v1[:,2:3]-v2[:,2:3])%math.tau
    dPhi21 = (v2[:,2:3]-v1[:,2:3])%math.tau
    dPhi = torch.min(dPhi12,dPhi21)
    return dPhi

def calcDeltaR(v1, v2): # expects PtEtaPhiM representation
    dEta = (v1[:,1:2,:]-v2[:,1:2,:])
    dPhi = calcDeltaPhi(v1, v2)
    return (dEta ** 2 + dPhi ** 2).sqrt()

def setLeadingEtaPositive(batched_v) -> torch.Tensor: # expects [batch, feature, jet nb]
    etaSign = 1-2*(batched_v[:,1,0:1]<0).float() # -1 if eta is negative, +1 if eta is zero or positive
    batched_v[:,1,:] = etaSign * batched_v[:,1,:]
    return batched_v

def setLeadingPhiTo0(batched_v) -> torch.Tensor: # expects [batch, feature, jet nb]
    # set phi = 0 for the leading jet and rotate the event accordingly
    phi_ref = batched_v[:,2,0:1] # get the leading jet phi for each event
    batched_v[:,2,:] = batched_v[:,2,:] - phi_ref # rotate all phi to make phi_lead = 0
    batched_v[:,2,:][batched_v[:,2,:]>torch.pi] -= 2*torch.pi # retransform the phi that are > pi
    batched_v[:,2,:][batched_v[:,2,:]<-torch.pi] += 2*torch.pi # same for the phi that are < -pi
    return batched_v

def setSubleadingPhiPositive(batched_v) -> torch.Tensor: # expects [batch, feature, jet nb]
    batched_v[:,2,1:4][batched_v[:,2,1]<0] += torch.pi # starting from phi2, add pi to j2, j3, j4 if phi2 < 0
    batched_v[:,2,:][batched_v[:,2,:]>torch.pi] -= 2*torch.pi # retransform the phi that are > pi

    #phiSign = 1-2*(batched_v[:,2,1:2]<0).float() # -1 if phi2 is negative, +1 if phi2 is zero or positive
    #batched_v[:,2,1:4] = phiSign * batched_v[:,2,1:4]
    return batched_v

def deltaR_correction(dec_j) -> torch.Tensor:
    deltaPhi = calcDeltaPhi(dec_j[:,:,(0,2,0,1,0,1)], dec_j[:,:,(1,3,2,3,3,2)])
    deltaEta = calcDeltaEta(dec_j[:,:,(0,2,0,1,0,1)], dec_j[:,:,(1,3,2,3,3,2)])
    inputDeltaR_squared = deltaEta**2 + deltaPhi**2 # get the DeltaR squared between jets
    closest_pair_of_dijets_deltaR, closest_pair_of_dijets_indices = torch.topk(inputDeltaR_squared[:,0,:], k=2, largest=False)  # get the 2 minimum DeltaR and their indices
                                                                                                                                # here I use [:,0,:] because second dimension is feature, which can be suppressed once youre
                                                                                                                                # only dealing with DeltaR
    if torch.any(closest_pair_of_dijets_deltaR < 0.16): # if any of them has a squared deltaR < 0.16
        less_than_016 = torch.nonzero(torch.lt(closest_pair_of_dijets_deltaR, 0.16))    # gives the [event_idx, pairing_idx_idx]
                                                                                        # event_idx gives the number of the event in which theres a pairing with DeltaR < 0.4
                                                                                        # pairing_idx_idx gives 0 or 1, depending if the pairing with DeltaR < 0.4 is the smallest or the second smallest, respectively
                                                                                        # imagine that in the same event (call it number 65), both 01 and 23 dijets have DeltaR < 0.4, then you will get 65 twice in event_idx
                                                                                        # like this [..., [69, 0], [69, 1], ...]
                                                                                        # this 0 or 1 should be indexed in closest_pair_of_dijets_indices, which will give you the true pairing index (i.e. 0-5)
        event_idx = less_than_016[:, 0]
        pairing_idx_idx = less_than_016[:, 1]

        # now we get the true index 0-5 of the pairing (0: 01, 1: 23, 2: 02, 3: 13, 4: 03, 5: 12)
        pairing_idx = closest_pair_of_dijets_indices[event_idx, pairing_idx_idx]

        # clone the tensor to be modified keeping only the events that will be corrected
        dec_j_temp = dec_j[event_idx].clone()
        
        # get a [nb_events_to_be_modified, 1, 4] tensor, in which each of the elements of the list corresponds to jet's individual eta and phi that shall be corrected
        first_elements_to_modify = dec_j_temp[:,:,(0,2,0,1,0,1)][torch.arange(dec_j_temp.shape[0]), :, pairing_idx].unsqueeze(1)[:,:,1:3]
        second_elements_to_modify = dec_j_temp[:,:,(1,3,2,3,3,2)][torch.arange(dec_j_temp.shape[0]), :, pairing_idx].unsqueeze(1)[:,:,1:3]
        
        # get a [nb_events_to_be_modified, 1, 1] tensor that parametrizes the movement along the line that joins the two points (eta1, phi1) - (eta2, phi2)
        t = (0.4*(1+inputDeltaR_squared[event_idx, :, pairing_idx].sqrt()) / inputDeltaR_squared[event_idx, :, pairing_idx].sqrt()).unsqueeze(1)
        
        # this will displace the second elements along the line that joins them so that their separation is 0.16
        # modify eta
        second_elements_to_modify[:,:,0:1] = first_elements_to_modify[:,:,0:1] + t * (second_elements_to_modify[:,:,0:1] - first_elements_to_modify[:,:,0:1])
        
        # modify phi
        phi_diff = torch.min((second_elements_to_modify[:,:,1:2] - first_elements_to_modify[:,:,1:2])%math.tau, (first_elements_to_modify[:,:,1:2] - second_elements_to_modify[:,:,1:2])%math.tau)
        # if modifying phi2, the slope is marked by the sign from phi2 to phi1 
        phi_slope_sign = torch.sign(second_elements_to_modify[:,:,1:2] - first_elements_to_modify[:,:,1:2])
        second_elements_to_modify[:,:,1:2] = first_elements_to_modify[:,:,1:2] + t * phi_slope_sign * phi_diff

        # express all pi in the range (-pi, pi)
        second_elements_to_modify[second_elements_to_modify[:,0,1]< -torch.pi,:,1]   += 2*torch.pi
        second_elements_to_modify[second_elements_to_modify[:,0,1]> +torch.pi,:,1]   -= 2*torch.pi

        second_pairing_idx = torch.tensor([1,3,2,3,3,2], dtype = torch.int32)
        jet_idx = second_pairing_idx[pairing_idx]

        for i, idx in enumerate(event_idx):
            dec_j[idx, 1:3, jet_idx[i]] = second_elements_to_modify[i].squeeze()
    else:
        pass
    return dec_j



#
# Some different non-linear units
#
def SiLU(x): #SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)

def NonLU(x): #Pick the default non-Linear Unit
    return SiLU(x) # often slightly better performance than standard ReLU
    #return F.relu(x)
    #return F.rrelu(x, training=training)
    #return F.leaky_relu(x, negative_slope=0.1)
    #return F.elu(x)



#
# embed inputs in feature space
#
class Input_Embed(nn.Module):
    def __init__(self, dimension, device='cpu', symmetrize=True, return_masses = False):
        super(Input_Embed, self).__init__()
        self.d = dimension
        self.device = device
        self.symmetrize = symmetrize
        self.return_masses = return_masses

        # embed inputs to dijetResNetBlock in target feature space
        self.jet_embed     = Ghost_Batch_Norm(3, features_out=self.d, conv=True, name='jet embedder', device=self.device) # phi is relative to dijet, mass is zero in toy data. # 3 features -> 8 features
        self.jet_conv      = Ghost_Batch_Norm(self.d, conv=True, name='jet convolution', device = self.device)

        self.dijet_embed   = Ghost_Batch_Norm(4, features_out=self.d, conv=True, name='dijet embedder', device = self.device) # phi is relative to quadjet, # 4 features -> 8 features
        self.dijet_conv    = Ghost_Batch_Norm(self.d, conv=True, name='dijet convolution', device = self.device) 

        self.quadjet_embed = Ghost_Batch_Norm(3 if self.symmetrize else 4, features_out=self.d, conv=True, name='quadjet embedder', device = self.device) # phi is removed. # 3 features -> 8 features
        self.quadjet_conv  = Ghost_Batch_Norm(self.d, conv=True, name='quadjet convolution', device = self.device)

        #self.register_buffer('tau', torch.tensor(math.tau, dtype=torch.float))

    
        
    def data_prep(self, j):
        j = j.clone()# prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        j = j.view(-1,4,4)

        d, dPxPyPzE = addFourVectors(j[:,:,(0,2,0,1,0,1)], # 6 pixels
                                     j[:,:,(1,3,2,3,3,2)])

        q, qPxPyPzE = addFourVectors(d[:,:,(0,2,4)],
                                     d[:,:,(1,3,5)], 
                                     v1PxPyPzE = dPxPyPzE[:,:,(0,2,4)],
                                     v2PxPyPzE = dPxPyPzE[:,:,(1,3,5)])
        
        if self.return_masses:
            m2j = d[:, 3:4, :].clone()
            m4j = q[:, 3:4, :].clone()

        # take log of pt, mass variables which have long tails
        # j = PxPyPzE(j)
        # j = torch.log(1+j.abs())*j.sign()
        j[:,(0,3),:] = torch.log(1+j[:,(0,3),:])
        d[:,(0,3),:] = torch.log(1+d[:,(0,3),:])
        q[:,(0,3),:] = torch.log(1+q[:,(0,3),:])

        # set up all possible jet pairings
        j = torch.cat([j, j[:,:,(0,2,1,3)], j[:,:,(0,3,1,2)]],2)

        if self.symmetrize:
            # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
            j[:,2:3,(0,2,4,6,8,10)] = calcDeltaPhi(d, j[:,:,(0,2,4,6,8,10)]) # replace jet phi with deltaPhi between dijet and jet
            j[:,2:3,(1,3,5,7,9,11)] = calcDeltaPhi(d, j[:,:,(1,3,5,7,9,11)])
        
            d[:,2:3,(0,2,4)] = calcDeltaPhi(q, d[:,:,(0,2,4)])
            d[:,2:3,(1,3,4)] = calcDeltaPhi(q, d[:,:,(1,3,5)])

            q = torch.cat( (q[:,:2,:],q[:,3:,:]) , 1 ) # remove phi from quadjet features

        if self.return_masses:
            return j, d, q, m2j, m4j
        else:
            return j, d, q

    def set_mean_std(self, j):
        if self.return_masses:
            j, d, q, _, _ = self.data_prep(j)
        else:
            j, d, q = self.data_prep(j)

        self    .jet_embed.set_mean_std(j[:,0:3])#mass is always zero in toy data
        self  .dijet_embed.set_mean_std(d)
        self.quadjet_embed.set_mean_std(q)

    def set_ghost_batches(self, n_ghost_batches):
        self.    jet_embed.set_ghost_batches(n_ghost_batches)
        self.  dijet_embed.set_ghost_batches(n_ghost_batches)
        self.quadjet_embed.set_ghost_batches(n_ghost_batches)

        self.    jet_conv.set_ghost_batches(n_ghost_batches)
        self.  dijet_conv.set_ghost_batches(n_ghost_batches)
        self.quadjet_conv.set_ghost_batches(n_ghost_batches)

    def forward(self, j):
        if self.return_masses:
            j, d, q, m2j, m4j = self.data_prep(j)
        else:
            j, d, q = self.data_prep(j)

        j = self    .jet_embed(j[:,0:3])#mass is always zero in toy data
        d = self  .dijet_embed(d)
        q = self.quadjet_embed(q)

        j = self    .jet_conv(NonLU(j))
        d = self  .dijet_conv(NonLU(d))
        q = self.quadjet_conv(NonLU(q))
        
        if self.return_masses:
            return j, d, q, m2j, m4j
        else:
            return j, d, q



class Basic_CNN(nn.Module):
    def __init__(self, dimension, n_classes=2, device='cpu'):
        super(Basic_CNN, self).__init__()
        self.device = device
        self.d = dimension
        self.n_classes = n_classes
        self.n_ghost_batches = 64

        self.name = f'Basic_CNN_{self.d}'

        self.input_embed = Input_Embed(self.d)

        self.jets_to_dijets     = Ghost_Batch_Norm(self.d, stride=2, conv=True, device = self.device)
        self.dijets_to_quadjets = Ghost_Batch_Norm(self.d, stride=2, conv=True, device = self.device)

        self.select_q = Ghost_Batch_Norm(self.d, features_out=1, conv=True, bias=False, device = self.device)
        self.out      = Ghost_Batch_Norm(self.d, features_out=self.n_classes, conv=True, device = self.device)

    def set_mean_std(self, j):
        self.input_embed.set_mean_std(j)

    def set_ghost_batches(self, n_ghost_batches):
        self.input_embed.set_ghost_batches(n_ghost_batches)
        self.jets_to_dijets.set_ghost_batches(n_ghost_batches)
        self.dijets_to_quadjets.set_ghost_batches(n_ghost_batches)
        self.select_q.set_ghost_batches(n_ghost_batches)
        self.out.set_ghost_batches(n_ghost_batches)
        self.n_ghost_batches = n_ghost_batches

    def forward(self, j):
        j, d, q = self.input_embed(j)

        d = d + NonLU(self.jets_to_dijets(j))
        q = q + NonLU(self.dijets_to_quadjets(d))

        #compute a score for each event quadjet
        q_logits = self.select_q(q)

        #convert the score to a 'probability' with softmax. This way the classifier is learning which pairing is most relevant to the classification task at hand.
        q_score = F.softmax(q_logits, dim=-1)
        q_logits = q_logits.view(-1, 3)

        #add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1,2))

        #project the final event-level pixel into the class score space
        c_logits = self.out(e)
        c_logits = c_logits.view(-1, self.n_classes)

        return c_logits, q_logits

class Basic_encoder(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, permute_input_jet = False, phi_rotations = False, return_masses = False, n_ghost_batches = -1, device = 'cpu'):
        super(Basic_encoder, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.permute_input_jet = permute_input_jet
        self.phi_rotations = phi_rotations
        self.return_masses = return_masses
        self.n_ghost_batches = n_ghost_batches
        

        self.name = f'Basic_encoder_{self.d_bottleneck}'

        self.input_embed            = Input_Embed(self.d, symmetrize=self.phi_rotations, return_masses=self.return_masses)
        self.jets_to_dijets         = Ghost_Batch_Norm(self.d, stride=2, conv=True, device=self.device)
        self.dijets_to_quadjets     = Ghost_Batch_Norm(self.d, stride=2, conv=True, device=self.device)
        self.select_q               = Ghost_Batch_Norm(self.d, features_out=1, conv=True, bias=False, device=self.device)

        self.bottleneck_in          = Ghost_Batch_Norm(self.d, features_out=self.d_bottleneck, conv=True, device=self.device)
        
    def set_mean_std(self, j):
        self.input_embed.set_mean_std(j)
  
    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = n_ghost_batches
        # encoder
        self.input_embed.set_ghost_batches(n_ghost_batches)
        self.jets_to_dijets.set_ghost_batches(n_ghost_batches)
        self.dijets_to_quadjets.set_ghost_batches(n_ghost_batches)
        self.select_q.set_ghost_batches(n_ghost_batches)
        # bottleneck_in
        self.bottleneck_in.set_ghost_batches(n_ghost_batches)

    def forward(self, j):
        #
        # Preparation block
        #
        j_rot = j.clone() # j.shape = [batch_size, 4, 4]
        
        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane
        j_rot = setSubleadingPhiPositive(setLeadingPhiTo0(setLeadingEtaPositive(j_rot))) if self.phi_rotations else j_rot
        
        if self.permute_input_jet: 
            for i in range(j.shape[0]): # randomly permute the input jets positions# randomly permute the input jets positions
                j_rot[i] = j[i, :, torch.randperm(4)]

        # convert to PxPyPzE and compute means and variances
        jPxPyPzE = PxPyPzE(j_rot) # j_rot.shape = [batch_size, 4, 4]

        #
        # Encode Block
        #
        if self.return_masses:
            j, d, q, m2j, m4j = self.input_embed(j_rot)                                         # j.shape = [batch_size, 8, 12] -> 12 = 0 1 2 3 0 2 1 3 0 3 1 2       
        else:
            j, d, q = self.input_embed(j_rot)                                                   # j.shape = [batch_size, 8, 12] -> 12 = 0 1 2 3 0 2 1 3 0 3 1 2
                                                                                                # d.shape = [batch_size, 8, 6]  -> 6 = 01 23 02 13 03 12
                                                                                                # q.shape = [batch_size, 8, 3]  -> 3 = 0123 0213 0312; 3 pixels each with 8 features
        d = d + NonLU(self.jets_to_dijets(j))                                                   # d.shape = [batch_size, 8, 6]
        q = q + NonLU(self.dijets_to_quadjets(d))                                               # q.shape = [batch_size, 8, 3]
        # compute a score for each event quadjet
        q_logits = self.select_q(q)                                                             # q_logits.shape = [batch_size, 1, 3] -> 3 = 0123 0213 0312
        # convert the score to a 'probability' with softmax. This way the classifier is learning which pairing is most relevant to the classification task at hand.
        q_score = F.softmax(q_logits, dim=-1)                                                   # q_score.shape = [batch_size, 1, 3]
        q_logits = q_logits.view(-1, 3)                                                         # q_logits.shape = [batch_size, 3, 1]
        # add together the quadjets with their corresponding probability weight
        e_in = torch.matmul(q, q_score.transpose(1,2))                                             # e.shape = [batch_size, 8, 1] (8x3 · 3x1 = 8x1)



        #
        # Bottleneck
        #
        z = NonLU(self.bottleneck_in(e_in))

        if self.return_masses:
            return jPxPyPzE, j_rot, z, m2j, m4j
        else:
            return jPxPyPzE, j_rot, z
   
class Basic_decoder(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, correct_DeltaR = False, return_masses = False, n_ghost_batches = -1, device = 'cpu'):
        super(Basic_decoder, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.correct_DeltaR = correct_DeltaR
        self.return_masses = return_masses
        self.n_ghost_batches = n_ghost_batches
        

        self.name = f'Basic_decoder_{self.d_bottleneck}'

        self.bottleneck_out         = Ghost_Batch_Norm(self.d_bottleneck, features_out=self.d, conv=True, device=self.device)

        self.decode_q               = Ghost_Batch_Norm(self.d, features_out=self.d, stride=3, conv_transpose=True, device=self.device)
        self.dijets_from_quadjets   = Ghost_Batch_Norm(self.d, features_out=self.d, stride=2, conv_transpose=True, device=self.device)
        self.jets_from_dijets       = Ghost_Batch_Norm(self.d, features_out=self.d, stride=2, conv_transpose=True, device=self.device)
        self.select_dec             = Ghost_Batch_Norm(self.d*4, features_out=1, conv=True, bias=False, device=self.device)

        self.jets_res_1 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.jets_res_2 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.jets_res_3 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.decode_j = Ghost_Batch_Norm(self.d, features_out=3, conv=True, device=self.device)
        
        self.decode_j = Ghost_Batch_Norm(self.d, features_out=4, conv=True, device=self.device)
        # self.expand_j = Ghost_Batch_Norm(self.d, features_out=128, conv=True, device=self.device)
        # self.decode_j = Ghost_Batch_Norm(128, features_out=3, conv=True, device=self.device)# jet mass is always zero, let's take advantage of this!

        # self.decode_1 = Ghost_Batch_Norm(  self.d, features_out=2*self.d, stride=2, conv_transpose=True, device=self.device)
        # self.decode_2 = Ghost_Batch_Norm(2*self.d, features_out=4*self.d, stride=2, conv_transpose=True, device=self.device)
        # self.decode_3 = Ghost_Batch_Norm(4*self.d, features_out=3,                  conv=True,           device=self.device)

    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = n_ghost_batches

        # bottleneck_out
        self.bottleneck_out.set_ghost_batches(n_ghost_batches)
        # decoder
        self.decode_q.set_ghost_batches(n_ghost_batches)
        self.dijets_from_quadjets.set_ghost_batches(n_ghost_batches)
        self.jets_from_dijets.set_ghost_batches(n_ghost_batches)
        self.select_dec.set_ghost_batches(n_ghost_batches)
        self.jets_res_1.set_ghost_batches(n_ghost_batches)
        # self.jets_res_2.set_ghost_batches(n_ghost_batches)
        # self.jets_res_3.set_ghost_batches(n_ghost_batches)
        # self.expand_j.set_ghost_batches(n_ghost_batches)
        self.decode_j.set_ghost_batches(n_ghost_batches)
  
    def forward(self, z):
        #
        # Bottleneck
        #
        e_out = NonLU(self.bottleneck_out(z))



        #
        # Decode Block
        #
        '''
        dec_d = NonLU(self.decode_1(e))     # 1 pixel to 2
        dec_j = NonLU(self.decode_2(dec_d)) # 2 pixel to 4
        dec_j =       self.decode_3(dec_j)  # down to four features per jet. Nonlinearity is sinh, cosh activations below
        '''
        dec_q = NonLU(self.decode_q(e_out))                                                     # dec_q.shape = [batch_size, 8, 3] 0123 0213 0312
        dec_d = NonLU(self.dijets_from_quadjets(dec_q))                                         # dec_d.shape = [batch_size, 8, 6] 01 23 02 13 03 12
        dec_j = NonLU(self.jets_from_dijets(dec_d))                                             # dec_j.shape = [batch_size, 8, 12]; dec_j is interpreted as jets 0 1 2 3 0 2 1 3 0 3 1 2
        
        dec_j = dec_j.view(-1, self.d, 3, 4)                                                    # last index is jet
        dec_j = dec_j.transpose(-1, -2)                                                         # last index is pairing history now 
        dec_j = dec_j.contiguous().view(-1, self.d * 4, 3)                                      # 32 numbers corresponding to each pairing: which means that you have 8 numbers corresponding to each jet in each pairing concatenated along the same dimension
                                                                                                # although this is not exact because now the information between jets is mixed, but thats the idea
        dec_j_logits = self.select_dec(dec_j)
        dec_j_score = F.softmax(dec_j_logits, dim = -1)                                         # 1x3

        dec_j = torch.matmul(dec_j, dec_j_score.transpose(1, 2))                                # (32x3 · 3x1 = 32x1)
        dec_j = dec_j.view(-1, self.d, 4)                                                       # 8x4

        # conv kernel 1
        j_res = dec_j.clone()
        dec_j = NonLU(self.jets_res_1(dec_j)) + j_res
        # j_res = dec_j.clone()
        # dec_j = NonLU(self.jets_res_2(dec_j))+j_res
        # j_res = dec_j.clone()
        # dec_j = NonLU(self.jets_res_3(dec_j))+j_res        
        # dec_j = self.expand_j(dec_j)
        # dec_j = NonLU(dec_j)
        dec_j = self.decode_j(dec_j)                                                            # 4x4
        

        # apply the DeltaR correction (in inference) so that jets are separated at least deltaR = 0.4
        dec_j = deltaR_correction(dec_j) if self.correct_DeltaR and not self.training else dec_j

        
        Pt = dec_j[:,0:1].cosh()+39 # ensures pt is >=40 GeV
        Eta = dec_j[:,1:2]
        Phi = dec_j[:,2:3]
        # M  = dec_j[:,3:4].cosh()-1 # >=0, in our case it is always zero for the toy data. we could relax this for real data
        M = dec_j[:,3:4].cosh()-1

        rec_j = torch.cat((Pt, Eta, Phi, M), 1)
        if self.return_masses:
            rec_d, rec_dPxPyPzE = addFourVectors(   rec_j[:,:,(0,2,0,1,0,1)], 
                                                    rec_j[:,:,(1,3,2,3,3,2)])
            rec_q, rec_qPxPyPzE = addFourVectors(   rec_d[:,:,(0,2,4)],
                                                    rec_d[:,:,(1,3,5)])
            rec_m2j = rec_d[:, 3:4, :].clone()
            rec_m4j = rec_q[:, 3:4, :].clone()

        Px = Pt*Phi.cos()
        Py = Pt*Phi.sin()
        Pz = Pt*Eta.sinh()

        
        # E  = (Pt**2+Pz**2+M**2).sqrt()   # ensures E^2>=M^2
        E  = (Pt**2+Pz**2).sqrt() # ensures E^2>=0. In our case M is zero so let's not include it
        
        rec_jPxPyPzE = torch.cat((Px, Py, Pz, E), 1)

        # # Nonlinearity for final output four-vector components
        # rec_jPxPyPzE = torch.cat((dec_j[:,0:3,:].sinh(), dec_j[:,3:4,:].cosh()), dim=1)
        if self.return_masses:
            return rec_jPxPyPzE, rec_j, z, rec_m2j, rec_m4j
        else:
            return rec_jPxPyPzE, rec_j, z

class Basic_CNN_AE(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, permute_input_jet = False, phi_rotations = False, correct_DeltaR = False, return_masses = False, device = 'cpu'):
        super(Basic_CNN_AE, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.permute_input_jet = permute_input_jet
        self.phi_rotations = phi_rotations
        self.correct_DeltaR = correct_DeltaR
        self.return_masses = return_masses
        self.n_ghost_batches = 64
        

        self.name = f'Basic_CNN_AE_{self.d_bottleneck}'

        self.encoder = Basic_encoder(dimension = self.d, bottleneck_dim = self.d_bottleneck, permute_input_jet = self.permute_input_jet, phi_rotations = self.phi_rotations, return_masses = self.return_masses, n_ghost_batches = self.n_ghost_batches, device = self.device)
        self.decoder = Basic_decoder(dimension = self.d, bottleneck_dim = self.d_bottleneck, correct_DeltaR = self.correct_DeltaR, return_masses = self.return_masses, n_ghost_batches = self.n_ghost_batches, device = self.device)

    
    def set_mean_std(self, j):
        self.encoder.set_mean_std(j)
    
    def set_ghost_batches(self, n_ghost_batches):
        self.encoder.set_ghost_batches(n_ghost_batches)
        self.decoder.set_ghost_batches(n_ghost_batches)

    
    def forward(self, j):
        #
        # Encode
        #
        if self.return_masses:
            jPxPyPzE, j_rot, z, m2j, m4j = self.encoder(j)      
        else:
            jPxPyPzE, j_rot, z = self.encoder(j)   
        
        #
        # Decode
        #
        if self.return_masses:
            rec_jPxPyPzE, rec_j, z, rec_m2j, rec_m4j = self.decoder(z)      
        else:
            rec_jPxPyPzE, rec_j, z = self.decoder(z)   



        if self.return_masses:
            return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j, z, m2j, m4j, rec_m2j, rec_m4j
        else:
            return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j, z



class K_Fold(nn.Module):
    def __init__(self, networks, task = 'FvT'):
        super(K_Fold, self).__init__()
        self.networks = networks
        for network in self.networks:
            network.eval()
        self.task = task

    @torch.no_grad()
    def forward(self, j, e):

        if self.task == 'SvB' or self.task == 'FvT': # i.e. if task is classification
            c_logits = torch.zeros(j.shape[0], self.networks[0].n_classes)
            q_logits = torch.zeros(j.shape[0], 3)

            for offset, network in enumerate(self.networks):
                mask = (e==offset)
                c_logits[mask], q_logits[mask] = network(j[mask])

            # shift logits to have mean zero over quadjets/classes. Has no impact on output of softmax, just makes logits easier to interpret
            c_logits = c_logits - c_logits.mean(dim=-1, keepdim=True)
            q_logits = q_logits - q_logits.mean(dim=-1, keepdim=True)

            return c_logits, q_logits  
        elif self.task == 'dec':
            rec_j = torch.zeros(j.shape[0], 4, 4)
            z = torch.zeros(j.shape[0], self.networks[0].d_bottleneck, 1)
            for offset, network in enumerate(self.networks):
                mask = (e==offset)
                if network.return_masses:
                    _, _, _, rec_j[mask], z[mask], _, _, _, _ = network(j[mask])
                else:
                    _, _, _, rec_j[mask], z[mask] = network(j[mask])
            return rec_j, z # save only j in PtEtaPhiM representation
        elif self.task == 'gen':
            rec_j = torch.zeros(j.shape[0], 4, 4)
            for offset, network in enumerate(self.networks):
                mask = (e==offset)
                z_offset = j[mask]
                z_sampled = GMM_sample(z_offset, max_nb_gaussians = 5, debug = True,  sample = 'fourTag_10x', density = True, offset = offset)
                if network.return_masses:
                    _, rec_j[mask], _, _, _ = network(z_sampled)
                else:
                    _, rec_j[mask], _ = network(z_sampled)
            return rec_j # save only j in PtEtaPhiM representation
        
        else:
            pass


def GMM_sample(z, max_nb_gaussians = 2, debug = False, **kwargs):
            dimension = z.shape[1]
            z = z.numpy()
            z_sampled = np.zeros_like(z) # create the final sampled activations
            if debug:
                sample = kwargs.get('sample', 'fourTag_10x')
                offset = kwargs.get('offset', None)
                density = kwargs.get("density", True) # default True
                import matplotlib.pyplot as plt
                # create necessary things to plot
                # Determine the grid layout based on the number of features
                if dimension <= 4:
                    num_rows = 2
                    num_cols = 2
                elif dimension <= 6:
                    num_rows = 2
                    num_cols = 3
                elif dimension <= 9:
                    num_rows = 3
                    num_cols = 3 
                elif dimension <= 12:
                    num_rows = 3
                    num_cols = 4
                elif dimension <= 16:
                    num_rows = 4
                    num_cols = 4
                else:
                    raise ValueError("The number of features is too high to display in a reasonable way.")
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20,8))
                axs = axs.flatten()
                if (dimension < num_rows * num_cols):
                    for j in range(1, num_rows*num_cols - dimension + 1):
                        axs[-j].axis('off')  # Hide any empty subplots
                h, bins = np.zeros_like(axs), np.zeros_like(axs)

            
            for d in range(dimension):
                min_bic = 0
                counter = 1
                gmms, fits, bics = [], [], []
                if offset is not None: print(f'\nOffset {offset}:')
                print(f'Running GMM with max. {max_nb_gaussians} gaussians for {d}-th feature')
                for i in range (max_nb_gaussians): # test the AIC/BIC metric between 1 and max_nb_gaussians components
                    gmm = GMM(n_components = counter, random_state=10, covariance_type = 'full')
                    gmms.append(gmm)
                    fits.append(gmm.fit(z[:,d]))
                    #labels = fit.predict(z[:,0])
                    bic = gmm.bic(z[:,d])
                    bics.append(bic)
                    if bic < min_bic or min_bic == 0:
                        min_bic = bic
                        n_opt = counter
                    counter = counter + 1
                # get optimal GMM model
                gmm_opt = gmms[n_opt - 1]
                # get optimal parameters
                means_opt = fits[n_opt - 1].means_
                covs_opt  = fits[n_opt - 1].covariances_
                weights_opt = fits[n_opt - 1].weights_

                if debug:
                    h[d], bins[d], _ = axs[d].hist(z[:,d], density=density, color='black', bins=32, histtype = 'step', lw = 3, label = 'True $z$')
                    density_factor = np.sum(h[d]*(bins[d][1:]-bins[d][:-1])) if not density else 1.
                    x_ax = np.linspace(bins[d][0], bins[d][-1], 1000)
                    y_axs = []
                    for i in range(n_opt):
                        y_axs.append(density_factor*norm.pdf(x_ax, float(means_opt[i][0]), np.sqrt(float(covs_opt[i][0][0])))*weights_opt[i]) # ith gaussian
                        axs[d].plot(x_ax, y_axs[i], lw = 1.5)
                        axs[d].tick_params(which = 'major', axis = 'both', direction='out', length = 6, labelsize = 20)
                        axs[d].minorticks_on()
                        axs[d].tick_params(which = 'minor', axis = 'both', direction='in', length = 3)
                    axs[d].plot(x_ax, np.sum(y_axs, axis = 0), lw = 1.5, ls='dashed', label = "GMM estimated PDF")
                    axs[d].set_title(f'Feature {d+1}; opt. comp. = {n_opt}', fontsize = 20)
                    axs[d].set_xlabel("Value", fontsize = 20)
                    if not density:
                        axs[d].set_ylabel(f"Counts / {(bins[d][1]-bins[d][0]):.1f}", fontsize = 20)
                    else:
                        axs[d].set_ylabel(f"Counts / {(bins[d][1]-bins[d][0]):.1f} (norm)", fontsize = 20)


                # Sampling
                r_values = np.random.uniform(0, 1, len(z[:,d]))
                # Cumulative sum of weights to sample the identity of the gaussian
                weights_cum = np.cumsum(weights_opt)
                # Find the indices of the values in weights_cumulative that are immediately higher than 'r'
                gaussian_indices = np.searchsorted(weights_cum, r_values[:, np.newaxis], side='right')[:,0]
                # Use list comprehension to get the parameters for the corresponding Gaussian distributions
                mu = [float(means_opt[i][0]) for i in gaussian_indices]
                sigma = [np.sqrt(float(covs_opt[i][0][0])) for i in gaussian_indices]

                # Sample from the corresponding Gaussian distributions
                z_sampled[:,d,0] = np.random.normal(mu, sigma)
                if debug:
                    axs[d].hist(z_sampled[:,d], bins = bins[d], color = "red", density = density, histtype = 'step', lw = 3, ls = 'solid', label = "Sampled $z$")
                    if d == 0:
                        fig.legend(loc='center', bbox_to_anchor=(0.25, 0.5), ncol=3, fontsize = 20)

            # layout
            fig.tight_layout()
            path = f"plots/redec/{sample}/"
            plots.mkpath(path)
            path_fig = f'{path}{sample}_zsampled_{dimension}_offset_{offset}.pdf' if offset is not None else f'{path}{sample}_zsampled_{dimension}.pdf'
            fig.savefig(path_fig)
            print(f'GMM plot saved to {path_fig}')      
            
            return torch.tensor(z_sampled)