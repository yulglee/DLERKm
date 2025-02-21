import torch
import torch.nn as nn
import torch.nn.functional as F

import esm

import pickle
from CBAM import AttentionNet

class ESM_2_pretrained(nn.Module):
    def __init__(self, dim, device='cuda', fine_tuning=True):
        super(ESM_2_pretrained, self).__init__()
        self.fine_tuning = fine_tuning
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(device=self.device)

        if self.fine_tuning == True:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False
    def forward(self, pro_seq):
        # data = [("protein", pro_seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(pro_seq)
        results = self.model(batch_tokens.to(device=self.device), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        return token_representations

class ERFPPK(nn.Module):
    def __init__(self, dim, device):
        super(ERFPPK, self).__init__()
        self.device = device
        self.ESM_2 = ESM_2_pretrained(dim=dim, device=self.device, fine_tuning=False)
        self.linear = nn.Sequential(nn.Linear(1280, dim),
                                    nn.LayerNorm(dim))
        self.reaction_feature = nn.Sequential(nn.Linear(dim, dim),
                                              nn.ReLU(),
                                              nn.Linear(dim, dim),
                                              nn.LayerNorm(dim))
        self.diff_fingerprints_encoding = nn.Linear(1024, dim)
        self.diff_layer_norm = nn.LayerNorm(dim)
        self.all_fingerprints_encoding = nn.Linear(1024, dim)
        self.all_layer_norm = nn.LayerNorm(dim)
        #Down stream
        self.cbam = AttentionNet(gate_channels=4, reduction_ratio=2, pool_types=['avg', 'max'])
        self.interact = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, 1))


    def forward(self, rs, Temperature, PH, Organism, E_sequence, differ_reaction_vector, all_reaction_vector, reaction_vectors):
        sub_finger_bina = differ_reaction_vector.to(self.device)
        pro_finger_bina = all_reaction_vector.to(self.device)
        sub_finger_feature = self.diff_fingerprints_encoding(sub_finger_bina)
        pro_finger_feature = self.all_fingerprints_encoding(pro_finger_bina)
        sf = self.diff_layer_norm(sub_finger_feature + pro_finger_feature)
        df = self.all_layer_norm(sub_finger_feature - pro_finger_feature)
        # ---------
        seq_vectors = self.linear(self.ESM_2(E_sequence)).squeeze(dim=0)
        if seq_vectors.dim() == 2:
            seq_vectors = seq_vectors.unsqueeze(0)

        seq_vectors = seq_vectors.permute(0, 2, 1)
        # 最大池化
        seq_vectors = F.max_pool1d(seq_vectors, kernel_size=seq_vectors.size(2))
        sv = seq_vectors.squeeze(-1)
        #---------------------------
        ev = self.reaction_feature(reaction_vectors)
        #--------------
        fv = torch.concatenate([ev.unsqueeze(dim=1), sv.unsqueeze(dim=1), sf.unsqueeze(dim=1), df.unsqueeze(dim=1)], dim=1)
        fv =self.cbam(fv)
        b, n, f = fv.shape
        out = self.interact(fv.reshape(b, n * f))
        return out.squeeze(dim=0)

#