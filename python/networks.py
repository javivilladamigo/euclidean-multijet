import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

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
    def __init__(self, dimension, device='cpu', symmetrize=True):
        super(Input_Embed, self).__init__()
        self.d = dimension
        self.device = device
        self.symmetrize = symmetrize

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

        return j, d, q

    def set_mean_std(self, j):
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
        j, d, q = self.data_prep(j)

        j = self    .jet_embed(j[:,0:3])#mass is always zero in toy data
        d = self  .dijet_embed(d)
        q = self.quadjet_embed(q)

        j = self    .jet_conv(NonLU(j))
        d = self  .dijet_conv(NonLU(d))
        q = self.quadjet_conv(NonLU(q))

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


class Basic_CNN_AE(nn.Module):
    def __init__(self, dimension, bottleneck_dim = None, phi_rotations = False, out_features = 12, device = 'cpu'):
        super(Basic_CNN_AE, self).__init__()
        self.device = device
        self.d = dimension
        self.d_bottleneck = bottleneck_dim if bottleneck_dim is not None else self.d
        self.out_features = out_features
        self.phi_rotations = phi_rotations
        self.n_ghost_batches = 64

        self.name = f'Basic_CNN_AE_{self.d_bottleneck}'

        self.input_embed            = Input_Embed(self.d, symmetrize=self.phi_rotations)
        self.jets_to_dijets         = Ghost_Batch_Norm(self.d, stride=2, conv=True, device=self.device)
        self.dijets_to_quadjets     = Ghost_Batch_Norm(self.d, stride=2, conv=True, device=self.device)
        self.select_q               = Ghost_Batch_Norm(self.d, features_out=1, conv=True, bias=False, device=self.device)

        self.bottleneck_in          = Ghost_Batch_Norm(self.d, features_out=self.d_bottleneck, conv=True, device=self.device)
        self.bottleneck_out         = Ghost_Batch_Norm(self.d_bottleneck, features_out=self.d, conv=True, device=self.device)

        self.decode_q               = Ghost_Batch_Norm(self.d, features_out=self.d, stride=3, conv_transpose=True, device=self.device)
        self.dijets_from_quadjets   = Ghost_Batch_Norm(self.d, features_out=self.d, stride=2, conv_transpose=True, device=self.device)
        self.jets_from_dijets       = Ghost_Batch_Norm(self.d, features_out=self.d, stride=2, conv_transpose=True, device=self.device)
        self.select_dec             = Ghost_Batch_Norm(self.d*4, features_out=1, conv=True, bias=False, device=self.device)

        self.jets_res_1 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.jets_res_2 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.jets_res_3 = Ghost_Batch_Norm(self.d, conv=True, device=self.device)
        # self.decode_j = Ghost_Batch_Norm(self.d, features_out=3, conv=True, device=self.device)
        
        self.decode_j = Ghost_Batch_Norm(self.d, features_out=3, conv=True, device=self.device)
        # self.expand_j = Ghost_Batch_Norm(self.d, features_out=128, conv=True, device=self.device)
        # self.decode_j = Ghost_Batch_Norm(128, features_out=3, conv=True, device=self.device)# jet mass is always zero, let's take advantage of this!

        # self.decode_1 = Ghost_Batch_Norm(  self.d, features_out=2*self.d, stride=2, conv_transpose=True, device=self.device)
        # self.decode_2 = Ghost_Batch_Norm(2*self.d, features_out=4*self.d, stride=2, conv_transpose=True, device=self.device)
        # self.decode_3 = Ghost_Batch_Norm(4*self.d, features_out=3,                  conv=True,           device=self.device)
        
        
        # self.decode_Px            = nn.Sequential(
        #     nn.Linear(in_features = 16, out_features = 32, device = self.device),
        #     nn.BatchNorm1d(32),
        #     nn.PReLU(),
        #     nn.Dropout(p = 0.2),
        #     #nn.Linear(in_features = 60, out_features = 70, device = self.device),
        #     #nn.Dropout(p = 0.1),
        #     #nn.Linear(in_features = 70, out_features = 80, device = self.device),
        #     #nn.Dropout(p = 0.1),
        #     #nn.PReLU(),
        #     #nn.Linear(in_features = 80, out_features = 85, device = self.device),
        #     #nn.PReLU(),
        #     nn.Linear(in_features = 32, out_features = 4, device = self.device),
        # )
        # self.decode_Py            = nn.Sequential(
        #     nn.Linear(in_features = 16, out_features = 32, device = self.device),
        #     nn.BatchNorm1d(32),
        #     nn.PReLU(),
        #     nn.Dropout(p = 0.2),
        #     #nn.Linear(in_features = 60, out_features = 70, device = self.device),
        #     #nn.Dropout(p = 0.1),
        #     #nn.Linear(in_features = 70, out_features = 80, device = self.device),
        #     #nn.Dropout(p = 0.1),
        #     #nn.PReLU(),
        #     #nn.Linear(in_features = 80, out_features = 85, device = self.device),
        #     #nn.PReLU(),
        #     nn.Linear(in_features = 32, out_features = 4, device = self.device),
        # )
        # '''
        # self.decode_PxPy            = nn.Sequential(
        #     nn.Linear(in_features = 16, out_features = 32, device = self.device),
        #     nn.BatchNorm1d(32),
        #     nn.PReLU(),
        #     nn.Dropout(p = 0.1),
        #     #nn.Linear(in_features = 60, out_features = 70, device = self.device),
        #     #nn.Dropout(p = 0.1),
        #     #nn.Linear(in_features = 70, out_features = 80, device = self.device),
        #     #nn.Dropout(p = 0.1),
        #     #nn.PReLU(),
        #     #nn.Linear(in_features = 80, out_features = 85, device = self.device),
        #     #nn.PReLU(),
        #     nn.Linear(in_features = 32, out_features = 8, device = self.device),
        # )
        # '''

        # self.decode_Pz             = nn.Sequential(
        #     nn.Linear(in_features = 16, out_features = 32, device = self.device),
        #     nn.BatchNorm1d(32),
        #     nn.PReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features = 32, out_features = 4, device = self.device),
        # )
        # self.decode_E               = nn.Sequential(
        #     nn.Linear(in_features = 16, out_features = 32, device = self.device),
        #     #nn.Dropout(p = 0.2),
        #     nn.BatchNorm1d(32),
        #     nn.PReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features = 32, out_features = 4, device = self.device),
        # )

        
    def set_mean_std(self, j):
        self.input_embed.set_mean_std(j)

    
    def set_ghost_batches(self, n_ghost_batches):
        self.n_ghost_batches = n_ghost_batches
        # encoder
        self.input_embed.set_ghost_batches(n_ghost_batches)
        self.jets_to_dijets.set_ghost_batches(n_ghost_batches)
        self.dijets_to_quadjets.set_ghost_batches(n_ghost_batches)
        self.select_q.set_ghost_batches(n_ghost_batches)
        # bottleneck
        self.bottleneck_in.set_ghost_batches(n_ghost_batches)
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

    
    def forward(self, j):
              # j.shape = [batch_size, 4, 4]
        #j[torch.all(j[:, 0]>40, axis = 1)] # pt > 30 GeV
        j_rot = j.clone()

        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane

        # maybe rotate the jets at the end, or take it out. Also it could be possible to rotate jets at the end to match the initial jets
        j_rot = setSubleadingPhiPositive(setLeadingPhiTo0(setLeadingEtaPositive(j_rot))) if self.phi_rotations else j_rot
        
        
        # remove and return from Input_Embed
        '''
        d_rot, dPxPyPzE_rot = addFourVectors(   j_rot[:,:,(0,2,0,1,0,1)], 
                                                        j_rot[:,:,(1,3,2,3,3,2)])
        q_rot, qPxPyPzE_rot = addFourVectors(   d_rot[:,:,(0,2,4)],
                                                        d_rot[:,:,(1,3,5)], 
                                                        v1PxPyPzE = dPxPyPzE_rot[:,:,(0,2,4)],
                                                        v2PxPyPzE = dPxPyPzE_rot[:,:,(1,3,5)])
        m2j = d_rot[:, 3:4, :]
        m4j = q_rot[:, 3:4, 0]
        '''

        # convert to PxPyPzE and compute means and variances
        jPxPyPzE = PxPyPzE(j_rot)
                                                                                                
                                                                                                # j_rot.shape = [batch_size, 4, 4]
        #
        # Encode Block
        #
        j, d, q = self.input_embed(j_rot)                                                       # j.shape = [batch_size, 8, 12] -> 12 = 0 1 2 3 0 2 1 3 0 3 1 2
                                                                                                # d.shape = [batch_size, 8, 6]  -> 6 = 01 23 02 13 03 12
                                                                                                # q.shape = [batch_size, 8, 3]  -> 3 = 0123 0213 0312; 3 pixels each with 8 features
        d = d + NonLU(self.jets_to_dijets(j))                                                   # d.shape = [batch_size, 8, 6]
        q = q + NonLU(self.dijets_to_quadjets(d))                                               # q.shape = [batch_size, 8, 3]
        #compute a score for each event quadjet
        q_logits = self.select_q(q)                                                             # q_logits.shape = [batch_size, 1, 3] -> 3 = 0123 0213 0312
        #convert the score to a 'probability' with softmax. This way the classifier is learning which pairing is most relevant to the classification task at hand.
        q_score = F.softmax(q_logits, dim=-1)                                                   # q_score.shape = [batch_size, 1, 3]
        q_logits = q_logits.view(-1, 3)                                                         # q_logits.shape = [batch_size, 3, 1]
        #add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1,2))                                             # e.shape = [batch_size, 8, 1] (8x3 · 3x1 = 8x1)

        #
        # Bottleneck
        #
        e = NonLU(self.bottleneck_in(e))
        e = NonLU(self.bottleneck_out(e))
        
        #
        # Decode Block
        #
        # dec_d = NonLU(self.decode_1(e))     # 1 pixel to 2
        # dec_j = NonLU(self.decode_2(dec_d)) # 2 pixel to 4
        # dec_j =       self.decode_3(dec_j)  # down to four features per jet. Nonlinearity is sinh, cosh activations below
        
        dec_q = NonLU(self.decode_q(e))                                                         # dec_q.shape = [batch_size, 8, 3] 0123 0213 0312
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
        dec_j = NonLU(self.jets_res_1(dec_j))+j_res
        # j_res = dec_j.clone()
        # dec_j = NonLU(self.jets_res_2(dec_j))+j_res
        # j_res = dec_j.clone()
        # dec_j = NonLU(self.jets_res_3(dec_j))+j_res        
        # dec_j = self.expand_j(dec_j)
        # dec_j = NonLU(dec_j)
        dec_j = self.decode_j(dec_j)                                                            # 4x4
        


        '''
        deltaPhi = calcDeltaPhi(dec_j[:,:,(0,2,0,1,0,1)], dec_j[:,:,(1,3,2,3,3,2)])
        deltaEta = calcDeltaEta(dec_j[:,:,(0,2,0,1,0,1)], dec_j[:,:,(1,3,2,3,3,2)])
        inputDeltaR_squared = deltaEta**2 + deltaPhi**2
        mask_DeltaR_below_threshold = (inputDeltaR_squared[:,:,:] < 0.16) # batch x 1 x 6
        deltaR_squared = 0.16 + F.relu(inputDeltaR_squared - 0.16) # this will output 0.16 for inputDeltaR_squared < 0.16 and inputDeltaR_squared for inputDeltaR_squared > 0.16

        # Create a new tensor that combines the original values and corrected values using the mask
        dec_j_new = dec_j.clone()
        # we'll keep deltaPhi fixed and correct the first eta of the pairing
        dec_j_new[:,1:2,(0,2,0,1,0,1)] = (1 - mask_DeltaR_below_threshold.float()) * dec_j[:,1:2,(0,2,0,1,0,1)] + mask_DeltaR_below_threshold.float()*dec_j[:,1:2,(1,3,2,3,3,2)] + (0.16 - (mask_DeltaR_below_threshold.float()*deltaPhi)**2).sqrt()
        dec_j = dec_j_new.clone()
        '''

        correct_DeltaR = True
        if correct_DeltaR:
            ### ALTERNATIVE METHOD ###
            deltaPhi = calcDeltaPhi(dec_j[:,:,(0,2,0,1,0,1)], dec_j[:,:,(1,3,2,3,3,2)])
            deltaEta = calcDeltaEta(dec_j[:,:,(0,2,0,1,0,1)], dec_j[:,:,(1,3,2,3,3,2)])
            inputDeltaR_squared = deltaEta**2 + deltaPhi**2 # get the DeltaR squared between jets
            closest_pair_of_dijets_deltaR, closest_pair_of_dijets_indices = torch.topk(inputDeltaR_squared[:,0,:], k=2, largest=False)  # get the 2 minimum DeltaR and their indeces
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

                pairing_idx = closest_pair_of_dijets_indices[event_idx, pairing_idx_idx] # now we get the true index 0-5 of the pairing (0: 01, 1: 23, 2: 02, 3: 13, 4: 03, 5: 12)

                dec_j_temp = dec_j[event_idx].clone() # clone the tensor to be modified keeping only the events that will be corrected
                
                
                # get a [nb_events_to_be_modified, 1, 4] tensor, in which each of the elements of the list corresponds to jet's individual eta and phi that shall be corrected
                first_elements_to_modify = dec_j_temp[:,:,(0,2,0,1,0,1)][torch.arange(dec_j_temp.shape[0]), :, pairing_idx].unsqueeze(1)[:,:,1:3]
                second_elements_to_modify = dec_j_temp[:,:,(1,3,2,3,3,2)][torch.arange(dec_j_temp.shape[0]), :, pairing_idx].unsqueeze(1)[:,:,1:3]
                
                t = (0.4*(1+torch.abs(torch.randn(1))*0.3) / inputDeltaR_squared[event_idx, :, pairing_idx].sqrt()).unsqueeze(1) # get a [nb_events_to_be_modified, 1, 1] tensor that parametrizes the movement along the line that joins the two points (eta1, phi1) - (eta2, phi2)
                
                # modify eta
                second_elements_to_modify[:,:,0:1] = first_elements_to_modify[:,:,0:1] + t * (second_elements_to_modify[:,:,0:1] - first_elements_to_modify[:,:,0:1]) # this will displace the second elements along the line that joins them so that their separation is 0.16
                
                #modify phi
                phi_diff = torch.min((second_elements_to_modify[:,:,1:2] - first_elements_to_modify[:,:,1:2])%math.tau, (first_elements_to_modify[:,:,1:2] - second_elements_to_modify[:,:,1:2])%math.tau)
                phi_slope_sign = torch.sign(second_elements_to_modify[:,:,1:2] - first_elements_to_modify[:,:,1:2]) # if modifying phi2, the slope is marked by the sign from phi2 to phi1 
                
                second_elements_to_modify[:,:,1:2] = first_elements_to_modify[:,:,1:2] + t * phi_slope_sign * phi_diff # this will displace the second elements along the line that joins them so that their separation is 0.16

                # express all pi in the range (-pi, pi)
                second_elements_to_modify[second_elements_to_modify[:,0,1]< -torch.pi,:,1]   += 2*torch.pi
                second_elements_to_modify[second_elements_to_modify[:,0,1]> +torch.pi,:,1]   -= 2*torch.pi

                second_pairing_idx = torch.tensor([1,3,2,3,3,2], dtype = torch.int32)
                jet_idx = second_pairing_idx[pairing_idx]

                for i, idx in enumerate(event_idx):
                    dec_j[idx, 1:3, jet_idx[i]] = second_elements_to_modify[i].squeeze()


        '''
        debug_deltaR_correction = False
        if debug_deltaR_correction:
            print("\ndec_j[1]") # element 1 has a DeltaR < 0.4 at pairing index 5 (pairing 12)
            print(dec_j[1])
            print("\ninputDeltaR_squared[1]")
            print(inputDeltaR_squared[1])
            print("\nmask")
            print(mask_DeltaR_below_threshold[1])

            print("\ndeltaR_squared[1]")
            print(deltaR_squared[1])

            print("DeltaEta, DeltaPhi:")
            print(deltaEta[1], deltaPhi[1])

            print("\ndeltaR_squared - deltaPhi")
            print((deltaR_squared - deltaPhi)[1])
            
            print("\ndec_j:")
            print(dec_j[1,1:2,(0,2,0,1,0,1)])
            print("\nOriginal eta:")
            print(dec_j[1,1:2])
            print("\nCorrected eta:")
            print(dec_j_new[1,1:2])
        else:
            pass
        '''
        
        Pt = dec_j[:,0:1].cosh()+39 # ensures pt is >=40 GeV
        Eta = dec_j[:,1:2]
        Phi = dec_j[:,2:3]
        M = dec_j[:,3:4].cosh()-1
        
        Px = Pt*Phi.cos()
        Py = Pt*Phi.sin()
        Pz = Pt*Eta.sinh()

        # M  =    dec_j[:,3:4].cosh()-1 # >=0, in our case it is always zero for the toy data. we could relax this for real data
        # E  = (Pt**2+Pz**2+M**2).sqrt()   # ensures E^2>=M^2
        E  = (Pt**2+Pz**2).sqrt() # ensures E^2>=0. In our case M is zero so let's not include it
        rec_j = torch.cat((Pt, Eta, Phi, M), 1)
        rec_jPxPyPzE = torch.cat((Px, Py, Pz, E), 1)
        
        # # Nonlinearity for final output four-vector components
        # rec_jPxPyPzE = torch.cat((dec_j[:,0:3,:].sinh(), dec_j[:,3:4,:].cosh()), dim=1)

        # using decode_* layers
        # dec_j = NonLU(dec_j) # NonLU ?                                           
        # rec_Px = self.decode_Px(dec_j.view(-1, 16)).view(-1, 1, 4)
        # rec_Py = self.decode_Py(dec_j.view(-1, 16)).view(-1, 1, 4)
        # rec_Pz = self.decode_Pz(dec_j.view(-1, 16)).view(-1, 1, 4)
        # rec_E = self.decode_E(dec_j.view(-1, 16)).view(-1, 1, 4)
        # rec_j = torch.cat((rec_Px, rec_Py, rec_Pz, rec_E), 1)

        # I would do this in the loss function
        # # produce all possible jet permutations for loss function
        # rec_jPxPyPzE = torch.zeros(*rec_j.shape, 24)
        # for k, perm in enumerate(list(itertools.permutations([0,1,2,3]))):
        #         rec_jPxPyPzE[:, :, :, k] = rec_j[:, :, perm] 


        # either do nothing or compute the 16 losses
        
        '''
        sorted_indices = PtEtaPhiM(rec_jPxPyPzE)[:,0,:].sort(descending = True, dim = 1)[1]
        for b in range(j.shape[0]):
            rec_jPxPyPzE[b, :, sorted_indices[b]]
        rec_jPxPyPzE[:, 1, 0] = 0 # leading Py = 0
        '''

        '''print("q:", q[0].data, dec_q[0].data)
        print("d:", d[0].data, dec_d[0].data)
        print("dec_j:", dec_j[0].data)
        print("sign:", sign_p[0].data)
        print("rec_jPxPyPzE:", rec_jPxPyPzE[0].data)'''

        
        '''
        rec_jPxPyPzE_sc = rec_jPxPyPzE.clone()
        for i in range(len(jPxPyPzE[0, :, 0])):
            # obtained a normalized j for the computation of the loss
            jPxPyPzE_sc[:, i, :] = (jPxPyPzE[:, i, :] - batch_mean[i]) / batch_std[i]
            rec_jPxPyPzE_sc[:, i, :] = (rec_jPxPyPzE[:, i, :] - batch_mean[i]) / batch_std[i]
        '''
        '''
        rec_d, rec_dPxPyPzE = addFourVectors(   PtEtaPhiM(rec_jPxPyPzE)[:,:,(0,2,0,1,0,1)], 
                                                PtEtaPhiM(rec_jPxPyPzE)[:,:,(1,3,2,3,3,2)])
        rec_q, rec_qPxPyPzE = addFourVectors(   rec_d[:,:,(0,2,4)],
                                                rec_d[:,:,(1,3,5)])
        rec_m2j = rec_d[:, 3:4, :]
        rec_m4j = rec_q[:, 3:4, 0]
        '''

        '''
        m2j_sc = m2j.clone()                                                                # [batch_size, 1, 6]
        m4j_sc = m4j.clone()                                                                # [batch_size, 1]
        
        m2j_mean = m2j.mean(dim = (0, 1))                                                       # [6]
        m4j_mean = m4j.mean(dim = 0)                                                            # [1]
        
        m2j_std = m2j.std(dim = (0, 1))                                                         # [6]
        m4j_std = m4j.std(dim = 0)                                                              # [1] 
        


        rec_m2j_sc  = rec_m2j.clone()
        rec_m4j_sc  = rec_m4j.clone()
        for i in range(len(rec_m2j[0, 0, :])):
            m2j_sc[:, 0, i] = (m2j_sc[:, 0, i] - m2j_mean[i]) / m2j_std[i]
            rec_m2j_sc[:, 0, i] = (rec_m2j[:, 0, i] - m2j_mean[i]) / m2j_std[i]
        '''            
        
        return jPxPyPzE, rec_jPxPyPzE, j_rot, rec_j





class VAE(nn.Module):
    def __init__(self, latent_dim, device = 'cpu'):
        super().__init__()
        self.device = device
        self.d = latent_dim
        self.enc_out_dim = self.d * 2
        self.n_ghost_batches = 0#16

        self.name = f'VAE_{self.d}'
        
        # encoder path
        self.encoder = Encoder(in_features=3, mult_factor=2, encoded_space_dim = self.enc_out_dim)

        # decoder path
        self.decode_q               = nn.ConvTranspose1d(self.d, self.d, 3)
        self.dijets_from_quadjets   = nn.ConvTranspose1d(self.d, self.d, 2, stride = 2)
        self.decode_d               = nn.ConvTranspose1d(self.d, self.d, 6)
        self.jets_from_dijets       = nn.ConvTranspose1d(self.d, self.d, 2, stride = 2)
        self.decode_j               = nn.ConvTranspose1d(self.d, self.d, 12)
        self.select_dec             = Ghost_Batch_Norm(self.d, features_out=1, conv=True, bias=False, device = self.device)


        # z distribution parameters
        self.fc_mu = nn.Linear(self.enc_out_dim, self.d)
        self.fc_var = nn.Linear(self.enc_out_dim, self.d)



        self.decode_Px              = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(in_features = 21, out_features = 100, device = self.device),
            nn.BatchNorm1d(100),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features = 100, out_features = 4, device = self.device),
            nn.Sigmoid()
        )
        self.decode_Py             = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(in_features = 21, out_features = 100, device = self.device),
            nn.BatchNorm1d(100),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features = 100, out_features = 4, device = self.device),
            nn.Tanh()

        )
        self.decode_Pz             = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(in_features = 21, out_features = 100, device = self.device),
            nn.BatchNorm1d(100),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features = 100, out_features = 4, device = self.device),
            nn.Hardtanh(min_val=-torch.pi, max_val=torch.pi)
        )
        self.decode_E               = nn.Sequential(
            nn.Flatten(start_dim = 1),
            nn.Linear(in_features = 21, out_features = 100, device = self.device),
            #nn.Dropout(p = 0.2),
            nn.BatchNorm1d(100),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features = 100, out_features = 4, device = self.device),
        )






        
        
        

    
    def forward(self, j):
        # j.shape = [batch_size, 4, 4]
        #j[torch.all(j[:, 0]>40, axis = 1)] # pt > 30 GeV
        j_rot = j.clone()

        # make leading jet eta positive direction so detector absolute eta info is removed
        # set phi = 0 for the leading jet and rotate the event accordingly
        # set phi1 > 0 by flipping wrt the xz plane
        # j_rot = setSubleadingPhiPositive(setLeadingPhiTo0(setLeadingEtaPositive(j_rot)))
        

        d_rot, dPxPyPzE_rot = addFourVectors(   j_rot[:,:,(0,2,0,1,0,1)], 
                                                        j_rot[:,:,(1,3,2,3,3,2)])
        q_rot, qPxPyPzE_rot = addFourVectors(   d_rot[:,:,(0,2,4)],
                                                        d_rot[:,:,(1,3,5)], 
                                                        v1PxPyPzE = dPxPyPzE_rot[:,:,(0,2,4)],
                                                        v2PxPyPzE = dPxPyPzE_rot[:,:,(1,3,5)])
        m2j = d_rot[:, 3:4, :]
        m4j = q_rot[:, 3:4, 0]


        e = self.encoder(j_rot[:,0:3,:]) # Energy (=mass) is not used as input
        e = e.view(-1, self.enc_out_dim)                                                       # [batch_dim, enc_out_dim, 1] --> [batch_dim, enc_out_dim] (typically enc_out_dim = 16)

        mu, log_var = self.fc_mu(e), self.fc_var(e)


        # sample z from Q(z|x)
        std = torch.exp(log_var / 2)
        #Q = torch.distributions.Normal(mu, std)
        #z = Q.rsample()
        mu, std = mu.view(-1, self.d, 1), std.view(-1, self.d, 1)
        z = mu + std
        #z = z.view(-1, self.d, 1)                                                          # recover [batch_dim, 8, 1] for decoding


        dec_q = self.decode_q(z)                                                                # dec_q.shape = [batch_size, 8, 3]
        dec_d =  self.decode_d(z) + NonLU(self.dijets_from_quadjets(dec_q))                     # dec_d.shape = [batch_size, 8, 6]
        dec_j = self.decode_j(z) + NonLU(self.jets_from_dijets(dec_d))                          # dec_j.shape = [batch_size, 8, 12]
        full_dec = torch.cat((dec_q, dec_d, dec_j), dim = 2)
        selected_dec = self.select_dec(full_dec)
        
        rec_Pt = torch.exp(2.5+ 5*self.decode_Px(selected_dec).view(-1, 1, 4))
        rec_Eta = 4 * self.decode_Py(selected_dec).view(-1, 1, 4)
        #rec_M = self.decode_E(selected_dec).view(-1, 1, 4)
        

        rec_Phi = self.decode_Pz(selected_dec).view(-1, 1, 4)






        #rec_j = (self.simple_dec3(self.simple_dec2(self.simple_dec1(selected_dec)))).view(-1, 4, 4)
        rec_j = torch.cat((rec_Pt, rec_Eta, rec_Phi), dim = 1)


        '''
        # normalization for loss
        rec_j_sc = rec_j.clone()
        for i in range(len(rec_j[0, :, 0])):
            # obtained a normalized j for the computation of the loss
            j_sc[:, i, :]   =     (j_rot[:, i, :] - batch_mean[i]) / batch_std[i]
            rec_j_sc[:, i, :] = (rec_j[:, i, :] - batch_mean[i]) / batch_std[i]



        e = self.encode(j_rot[:,0:3,:]) # Energy (=mass) is not used as input
        #print(e.shape)

        dec_q = self.decode_q(e)                                                                # dec_q.shape = [batch_size, 8, 3]
        dec_d =  self.decode_d(e) + NonLU(self.dijets_from_quadjets(dec_q))                     # dec_d.shape = [batch_size, 8, 6]
        dec_j = self.decode_j(e) + NonLU(self.jets_from_dijets(dec_d))                          # dec_j.shape = [batch_size, 8, 12]
        full_dec = torch.cat((dec_q, dec_d, dec_j), dim = 2)
        selected_dec = self.select_dec(full_dec)
        rec_Pt = self.decode_Px(selected_dec)
        rec_Eta = self.decode_Py(selected_dec)
        rec_Phi = self.decode_Pz(selected_dec)
        rec_E = self.decode_E(selected_dec)
        
        
        rec_j = torch.cat((rec_Pt, rec_Eta, rec_Phi, rec_E), dim = 2).permute(0,2,1)
        rec_j_sc = rec_j.clone()
        for i in range(len(j_rot[0, :, 0])):
            # obtained a normalized j for the computation of the loss
            j_sc[:, i, :]   =     (j_rot[:, i, :] - batch_mean[i]) / batch_std[i]
            rec_j_sc[:, i, :] = (rec_j[:, i, :] - batch_mean[i]) / batch_std[i]
        
        
        rec_d, rec_dPxPyPzE = addFourVectors(   rec_j[:,:,(0,2,0,1,0,1)], 
                                                rec_j[:,:,(1,3,2,3,3,2)])
        rec_q, rec_qPxPyPzE = addFourVectors(   rec_d[:,:,(0,2,4)],
                                                rec_d[:,:,(1,3,5)],
                                                v1PxPyPzE = rec_dPxPyPzE[:,:,(0,2,4)],
                                                v2PxPyPzE = rec_dPxPyPzE[:,:,(1,3,5)])
        rec_m2j = rec_d[:, 3:4, :]
        rec_m4j = rec_q[:, 3:4, 0]


        m2j_sc = m2j.clone()                                                                # [batch_size, 1, 6]
        m4j_sc = m4j.clone()                                                                # [batch_size, 1]
        
        m2j_mean = m2j.mean(dim = (0, 1))                                                       # [6]
        m4j_mean = m4j.mean(dim = 0)                                                            # [1]
        
        m2j_std = m2j.std(dim = (0, 1))                                                         # [6]
        m4j_std = m4j.std(dim = 0)                                                              # [1] 
        


        rec_m2j_sc  = rec_m2j.clone()
        rec_m4j_sc  = rec_m4j.clone()
        for i in range(len(rec_m2j[0, 0, :])):
            m2j_sc[:, 0, i] = (m2j_sc[:, 0, i] - m2j_mean[i]) / m2j_std[i]
            rec_m2j_sc[:, 0, i] = (rec_m2j[:, 0, i] - m2j_mean[i]) / m2j_std[i]
        '''
            
        
        return rec_j, z, mu, std
    
class K_Fold(nn.Module):
    def __init__(self, models, task = 'FvT'):
        super(K_Fold, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()
        self.task = task

    @torch.no_grad()
    def forward(self, j, e):

        if self.task == 'SvB' or self.task == 'FvT': # i.e. if task is classification
            c_logits = torch.zeros(j.shape[0], self.models[0].n_classes)
            q_logits = torch.zeros(j.shape[0], 3)

            for offset, model in enumerate(self.models):
                mask = (e==offset)
                c_logits[mask], q_logits[mask] = model(j[mask])

            # shift logits to have mean zero over quadjets/classes. Has no impact on output of softmax, just makes logits easier to interpret
            c_logits = c_logits - c_logits.mean(dim=-1, keepdim=True)
            q_logits = q_logits - q_logits.mean(dim=-1, keepdim=True)

            return c_logits, q_logits
        
        elif self.task == 'dec':
            rec_j = torch.zeros(j.shape[0], 3, 4) # [batch_size, 4jets, 4features]

            for offset, model in enumerate(self.models):
                #mask = (e==offset)
                rec_j, rec_j_sc, j_rot, j_sc, rec_m2j, m2j, rec_m4j, m4j = model(j)

            return rec_j



    
