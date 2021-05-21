import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np

class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x



class MIL_Attention_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2):
        super(MIL_Attention_fc, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
                
    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.attention_net = self.attention_net.to(device)
            self.classifier = self.classifier.to(device)
            self.device = device
        
        else:
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.attention_net = self.attention_net.to(device)
            self.classifier = self.classifier.to(device)
            self.device = None
        
    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict

class MIL_Attention_fc_surv(MIL_Attention_fc):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 4):
        super(MIL_Attention_fc_surv, self).__init__(gate = gate, size_arg = size_arg, dropout = dropout, n_classes = n_classes)

    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return hazards, S, Y_hat, A_raw, results_dict