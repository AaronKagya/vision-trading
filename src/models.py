"""Numerical and visual baselines; causal latent world-model implementation.

IMPORTANT: at decision time t, a forecast uses images ending at or BEFORE t.
A future frame t+1 may be used ONLY as an auxiliary target during fitting,
computed separately from the inference forward pass, never at inference.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LATENT = 64
ENC_OPTS = {'vit_res':224}
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN=(.485,.456,.406)
IMAGENET_STD=(.229,.224,.225)

class CNNEncoder(nn.Module):
    def __init__(self,out=LATENT):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2))
        self.net=nn.Sequential(block(3,16),block(16,32),block(32,64),block(64,64))
        self.pool=nn.AdaptiveAvgPool2d((4,4))
        self.fc=nn.Linear(64*4*4,out)
    def forward(self,x):
        return self.fc(self.pool(self.net(x)).flatten(1))

class ResNetEncoder(nn.Module):
    def __init__(self,pretrained=False,out=LATENT):
        super().__init__()
        import torchvision
        weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.m=torchvision.models.resnet18(weights=weights)
        self.m.fc=nn.Identity()
        self.fc=nn.Linear(512,out)
        self.pretrained=pretrained
        self.register_buffer('mean',torch.tensor(IMAGENET_MEAN).view(1,3,1,1),persistent=False)
        self.register_buffer('std',torch.tensor(IMAGENET_STD).view(1,3,1,1),persistent=False)
    def forward(self,x):
        if self.pretrained:x=((x+.5)-self.mean)/self.std
        return self.fc(self.m(x))

class ViTEncoder(nn.Module):
    def __init__(self,pretrained=False,out=LATENT):
        super().__init__()
        import timm
        self.res=ENC_OPTS['vit_res']
        self.m=timm.create_model('vit_tiny_patch16_224',pretrained=pretrained,num_classes=0,img_size=self.res)
        self.fc=nn.Linear(self.m.num_features,out)
        self.pretrained=pretrained
        self.register_buffer('mean',torch.tensor(IMAGENET_MEAN).view(1,3,1,1),persistent=False)
        self.register_buffer('std',torch.tensor(IMAGENET_STD).view(1,3,1,1),persistent=False)
    def forward(self,x):
        x=F.interpolate(x,size=(self.res,self.res),mode='bilinear',align_corners=False)
        if self.pretrained:x=((x+.5)-self.mean)/self.std
        return self.fc(self.m(x))

def make_encoder(name,pretrained=False):
    if name=='cnn':
        if pretrained:raise ValueError('CNN is not pretrained. Use --encoder resnet18 or vit_tiny.')
        return CNNEncoder()
    if name=='resnet18':return ResNetEncoder(pretrained)
    if name=='vit_tiny':return ViTEncoder(pretrained)
    raise ValueError(f'Unknown encoder {name}')

class WorldModel(nn.Module):
    def __init__(self,encoder,mode='wm',K=6):
        super().__init__()
        if mode not in ('static','wm_nolatent','wm'):raise ValueError(mode)
        self.enc=encoder;self.mode=mode;self.K=K
        self.ln=nn.LayerNorm(LATENT)
        self.gru=nn.GRU(LATENT,64,batch_first=True)
        self.pred=nn.Linear(64,LATENT)
        self.head=nn.Sequential(nn.Linear(LATENT,32),nn.ReLU(),nn.Linear(32,1))
    def embed(self,x):
        B,T=x.shape[:2]
        return self.ln(self.enc(x.flatten(0,1))).reshape(B,T,-1)
    def forward(self,x):
        n_expected=1 if self.mode=='static' else self.K
        if x.ndim!=5 or x.shape[1]!=n_expected or x.shape[2]!=3:
            raise ValueError(f'Expected exactly {n_expected} historical frames shaped B,T,3,H,W')
        z=self.embed(x)
        if self.mode=='static':return self.head(z[:,-1]).squeeze(-1),None,z
        h,_=self.gru(z)
        h=h[:,-1]
        return self.head(h).squeeze(-1),self.pred(h),z

class LSTMBaseline(nn.Module):
    def __init__(self,nin=12):
        super().__init__()
        self.lstm=nn.LSTM(nin,48,batch_first=True)
        self.head=nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1))
    def forward(self,x):
        h,_=self.lstm(x)
        return self.head(h[:,-1]).squeeze(-1)

def _img(imgs,idx):
    idx=np.asarray(idx,dtype=np.int64)
    if idx.size and (idx.min()<0 or idx.max()>=len(imgs)):
        raise IndexError('Out-of-range image indices (possible causality error)')
    return torch.as_tensor(imgs[idx],device=DEV,dtype=torch.float32).div_(255).sub_(.5)

def _set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def _val_loss(model,imgs,y,idx,offs,bs):
    model.eval();total=0.;count=0
    with torch.no_grad():
        for start in range(0,len(idx),bs):
            t=idx[start:start+bs]
            logits,_,_=model(_img(imgs,t[:,None]+offs[None,:]))
            labels=torch.as_tensor(y[t],device=DEV,dtype=torch.float32)
            total+=F.binary_cross_entropy_with_logits(logits,labels,reduction='sum').item()
            count+=len(t)
    return total/max(count,1)

def fit_world_model(imgs,y,tr,va,mode,K,enc_name,epochs,lr,bs,lam,seed,pretrained=False):
    """Train with strictly causal direction head; next image is training-only target."""
    if not len(tr) or not len(va):raise ValueError('Empty training or validation fold')
    _set_seed(seed);rng=np.random.default_rng(seed)
    model=WorldModel(make_encoder(enc_name,pretrained),mode,K).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    # Causal offsets only. No +1 is ever passed into the model's forward call.
    offs=np.array([0]) if mode=='static' else np.arange(1-K,1)
    yt=torch.as_tensor(y,device=DEV,dtype=torch.float32)
    best=float('inf');best_state=None;bad=0
    for epoch in range(epochs):
        model.train()
        perm=rng.permutation(tr)
        for start in range(0,len(tr),bs):
            b=perm[start:start+bs]
            if len(b)<2:continue
            x=_img(imgs,b[:,None]+offs[None,:]); logits,pred,z=model(x)
            loss=F.binary_cross_entropy_with_logits(logits,yt[b])
            if mode=='wm':
                # An auxiliary target at t+1 is legal ONLY during supervised
                # training. BN frozen, gradients disabled, and target NOT part
                # of prediction model input or its batch statistics.
                encoder_was_training=model.enc.training
                ln_was_training=model.ln.training
                model.enc.eval();model.ln.eval()
                with torch.no_grad():
                    target=model.ln(model.enc(_img(imgs,b+1))).detach()
                model.enc.train(encoder_was_training)
                model.ln.train(ln_was_training)
                loss=loss+lam*(1-F.cosine_similarity(pred,target,dim=-1)).mean()
                loss=loss+.1*F.relu(1-z.reshape(-1,LATENT).std(0,unbiased=False)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        val=_val_loss(model,imgs,y,va,offs,bs)
        if val<best-1e-4:
            best=val;bad=0;best_state=copy.deepcopy(model.state_dict())
        else:
            bad+=1
            if bad>=3:break
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    return model,offs

@torch.no_grad()
def predict_world_model(model,offs,imgs,idx,bs=256):
    offs=np.asarray(offs)
    if np.any(offs>0):raise ValueError('future frame offsets forbidden during prediction')
    model.eval();preds=[];latents=[]
    for start in range(0,len(idx),bs):
        b=np.asarray(idx[start:start+bs],dtype=np.int64)
        p,_,z=model(_img(imgs,b[:,None]+offs[None,:]))
        preds.append(torch.sigmoid(p).cpu().numpy())
        latents.append(z[:,-1].cpu().numpy())
    if not preds:return np.empty(0,dtype=np.float32),float('nan')
    zs=np.concatenate(latents)
    return np.concatenate(preds),float(zs.std(0).mean())

def fit_predict_lstm(num,y,tr,va,te,epochs,lr,bs,seed,checkpoint=None):
    _set_seed(seed);rng=np.random.default_rng(seed)
    sd=num[tr].reshape(-1,num.shape[-1]).std(0)+1e-9
    # Train-only scaler. No validation/test fit statistics.
    x=torch.as_tensor(num/sd,device=DEV,dtype=torch.float32)
    yt=torch.as_tensor(y,device=DEV,dtype=torch.float32)
    model=LSTMBaseline(num.shape[-1]).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');state=None;bad=0
    for epoch in range(epochs):
        model.train();perm=rng.permutation(tr)
        for start in range(0,len(perm),bs):
            b=perm[start:start+bs]
            if len(b)<2:continue
            loss=F.binary_cross_entropy_with_logits(model(x[b]),yt[b])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            val=F.binary_cross_entropy_with_logits(model(x[va]),yt[va]).item()
        if val<best-1e-4:
            best=val;state=copy.deepcopy(model.state_dict());bad=0
        else:
            bad+=1
            if bad>=3:break
    if state is not None:model.load_state_dict(state)
    if checkpoint:
        torch.save({'state_dict':model.state_dict(),'scaler_std':torch.as_tensor(sd).cpu(),'n_features':num.shape[-1]},checkpoint)
    model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(x[te])).cpu().numpy()
    del x
    return p


# Phase 2 candidate: representation fusion (new experimental configuration).
class FusionModel(nn.Module):
    """Same-candle numerical LSTM and single-chart CNN; no external information.

    Both branches see OHLC ending at the forecast decision time t. The image
    is deterministically rendered from those OHLC values, so this architecture
    tests a *representation bias*, not new market information.
    """
    def __init__(self, n_features=12):
        super().__init__()
        self.numeric = nn.LSTM(n_features, 48, batch_first=True)
        self.visual = CNNEncoder(out=64)
        self.head = nn.Sequential(
            nn.LayerNorm(48 + 64),
            nn.Linear(48 + 64, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
        )

    def forward(self, numeric, chart):
        if numeric.ndim != 3 or chart.ndim != 4 or chart.shape[1] != 3:
            raise ValueError('Expected numeric (batch,window,features), chart (batch,3,H,W)')
        if numeric.shape[0] != chart.shape[0]:
            raise ValueError('Numeric and chart batch sizes differ')
        state, _ = self.numeric(numeric)
        view = self.visual(chart)
        return self.head(torch.cat([state[:, -1, :], view], dim=1)).squeeze(-1)


def fit_predict_fusion(imgs, num, y, tr, va, te, epochs, lr, bs, seed, checkpoint=None):
    """Train fusion model with train-only numeric scaling, validation-only early stop.

    The forward pass uses only `imgs[t]` and `num[t]` at each decision t.
    `imgs[t+1]` is never accessed by this model, even while training.
    """
    if not len(tr) or not len(va) or not len(te):
        raise ValueError('Empty train, validation, or test partition')
    _set_seed(seed)
    rng = np.random.default_rng(seed)
    # The training split alone determines the transformation.
    flat = num[tr].reshape(-1, num.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-9
    numeric = torch.as_tensor((num - mean) / std, device=DEV, dtype=torch.float32)
    labels = torch.as_tensor(y, device=DEV, dtype=torch.float32)
    model = FusionModel(n_features=num.shape[-1]).to(DEV)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float('inf')
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        shuffled = rng.permutation(tr)
        for start in range(0, len(shuffled), bs):
            idx = shuffled[start:start + bs]
            if len(idx) < 2:
                continue
            logits = model(numeric[idx], _img(imgs, idx))
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(va), bs):
                idx = va[start:start + bs]
                logits = model(numeric[idx], _img(imgs, idx))
                total += F.binary_cross_entropy_with_logits(
                    logits, labels[idx], reduction='sum').item()
        value = total / len(va)
        if value < best_loss - 1e-4:
            best_loss, stale = value, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint:
        torch.save({'state_dict': model.cpu().state_dict(), 'scaler_mean': torch.as_tensor(mean).cpu(),
                    'scaler_std': torch.as_tensor(std).cpu(), 'n_features': num.shape[-1],
                    'architecture': 'causal_fusion_v1'}, checkpoint)
        model = model.to(DEV)
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(te), bs):
            idx = te[start:start + bs]
            logits = model(numeric[idx], _img(imgs, idx))
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)
