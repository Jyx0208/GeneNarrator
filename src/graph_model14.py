"""Small PyG pathway adapters followed by the archived GeneNarrator fusion."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import DenseGCNConv
from original_weibull_loss_corrected import weibull_nll

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_gn14',P/'original_round10/model_snapshot_0.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

class GraphAdapter(nn.Module):
    def __init__(self,adj):
        super().__init__();self.register_buffer('adj',torch.as_tensor(adj,dtype=torch.float32))
        self.identity=nn.Embedding(50,8);self.input=nn.Linear(9,32)
        self.convs=nn.ModuleList([DenseGCNConv(32,32),DenseGCNConv(32,32)])
        self.norms=nn.ModuleList([nn.LayerNorm(32),nn.LayerNorm(32)])
        self.drop=nn.Dropout(.2);self.attention=nn.Linear(32,1);self.output=nn.Linear(32,768)
    def forward(self,p):
        ident=self.identity.weight.unsqueeze(0).expand(p.shape[0],-1,-1)
        h=torch.nn.functional.gelu(self.input(torch.cat([p.unsqueeze(-1),ident],-1)))
        for conv,norm in zip(self.convs,self.norms):h=norm(h+self.drop(torch.nn.functional.gelu(conv(h,self.adj))))
        a=self.attention(h).softmax(dim=1)
        return self.output((a*h).sum(dim=1))

class TrialModel(nn.Module):
    def __init__(self,variant,adj):
        super().__init__();self.core=original.GeneNarratorAFT(gene_dim=1000,text_dim=768)
        self.adapter=(nn.Sequential(nn.Linear(50,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,768)) if variant=='MLP' else GraphAdapter(adj))
    def forward(self,gene,pathways):
        c=self.core;g=c.gene_encoder(gene);t=c.text_encoder(self.adapter(pathways))
        qg=c.gene_quality(g);qt=c.text_quality(t)
        ge=c.cross_attn_g2t(g.unsqueeze(1),t.unsqueeze(1)).squeeze(1)
        te=c.cross_attn_t2g(t.unsqueeze(1),g.unsqueeze(1)).squeeze(1)
        w=c.adaptive_gate(torch.cat([ge,te,qg,qt],dim=-1))
        param=c.weibull_head(w[:,0:1]*ge+w[:,1:2]*te)
        # Common numerical correction, declared before all trial arms are fitted.
        # Remove the archived ~4915-day hard scale ceiling; fusion/head layers are unchanged.
        scale=param[:,0].clamp(0.,14.).exp();shape=.5+3*param[:,1].sigmoid()
        return scale,shape,w,qg.flatten(),qt.flatten()

def loss(parts,time,event,regularize=True):
    scale,shape,w,qg,qt=parts;l=weibull_nll(scale,shape,time,event)
    if regularize:l=l+.01*((scale.log()-6.5)**2).mean()+.01*((shape-1.5)**2).mean()+.1*((qg-.5)**2).mean()+.1*((qt-.5)**2).mean()
    return l

def fit_preprocess(x,p,genes,*,gene_selection='pooled_variance',cohort=None,n_genes=1000):
    med=np.nanmedian(x,axis=0);filled=np.where(np.isfinite(x),x,med)
    if gene_selection == 'pooled_variance':
        variance = np.var(filled,axis=0)
    elif gene_selection == 'within_source_variance':
        cohort=np.asarray(cohort)
        if cohort.ndim != 1 or len(cohort) != len(x) or len(np.unique(cohort)) < 2:
            raise ValueError('Within-source selection requires aligned training cohort labels from at least two sources')
        variance=sum(np.sum(cohort==c)*np.var(filled[cohort==c],axis=0,dtype=np.float64)
                     for c in np.unique(cohort))/len(filled)
    else:
        raise ValueError(f'Unknown gene-selection policy: {gene_selection}')
    if not isinstance(n_genes,int) or n_genes <= 0:
        raise ValueError('n_genes must be a positive integer')
    ix=np.lexsort((genes,-variance))[:n_genes]
    return {'ix':ix,'median':med[ix],'gmean':filled[:,ix].mean(0),'gstd':np.maximum(filled[:,ix].std(0),1e-6),'pmean':p.mean(0),'pstd':np.maximum(p.std(0),1e-6)}

def transform(x,p,pp):
    g=x[:,pp['ix']];g=np.where(np.isfinite(g),g,pp['median'])
    return np.clip((g-pp['gmean'])/pp['gstd'],-5,5).astype(np.float32),np.clip((p-pp['pmean'])/pp['pstd'],-5,5).astype(np.float32)
