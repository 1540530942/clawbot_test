#!/usr/bin/env python3
# 第 10 期迷你 CLIP：python3 10_mini_clip.py 0 1500
# 单文件版：下面依次是极小的自动微分、迷你 Transformer 和实验本身，只依赖 NumPy。
import numpy as np
class T:
    def __init__(s,v,ps=(),bk=None):
        s.v=np.asarray(v,dtype=np.float64); s.g=None; s.ps=ps; s.bk=bk
    def __add__(s,o):
        o=o if isinstance(o,T) else T(o); r=T(s.v+o.v,(s,o))
        r.bk=lambda g:(unb(g,s.v.shape),unb(g,o.v.shape)); return r
    def __mul__(s,o):
        o=o if isinstance(o,T) else T(o); r=T(s.v*o.v,(s,o))
        r.bk=lambda g:(unb(g*o.v,s.v.shape),unb(g*s.v,o.v.shape)); return r
    def __matmul__(s,o):
        r=T(s.v@o.v,(s,o))
        r.bk=lambda g:(unb(g@np.swapaxes(o.v,-1,-2),s.v.shape),unb(np.swapaxes(s.v,-1,-2)@g,o.v.shape)); return r
    def t(s,*ax):
        r=T(s.v.transpose(ax),(s,)); inv=np.argsort(ax); r.bk=lambda g:(g.transpose(inv),); return r
    def rs(s,*sh):
        r=T(s.v.reshape(sh),(s,)); old=s.v.shape; r.bk=lambda g:(g.reshape(old),); return r
    def relu(s):
        m=s.v>0; r=T(s.v*m,(s,)); r.bk=lambda g:(g*m,); return r
    def softmax(s):
        x=s.v-s.v.max(-1,keepdims=True); e=np.exp(x); p=e/e.sum(-1,keepdims=True); r=T(p,(s,))
        def bk(g):
            gp=(g*p).sum(-1,keepdims=True); return (p*(g-gp),)
        r.bk=bk; return r
    def lnorm(s,eps=1e-6):
        # 只算方差归一化（无可学习缩放）
        m=s.v.mean(-1,keepdims=True); d=s.v-m; sd=np.sqrt((d*d).mean(-1,keepdims=True)+eps); r=T(d/sd,(s,))
        def bk(g):
            gc=g-g.mean(-1,keepdims=True); y=d/sd; gy=(g*y).mean(-1,keepdims=True)
            return ((gc-gy*y)/sd,)
        r.bk=bk; return r
    def backward(s,seed=None):
        order=[];seen=set()
        def go(n):
            if id(n) in seen:return
            seen.add(id(n))
            for p in n.ps: go(p)
            order.append(n)
        go(s); s.g=np.ones_like(s.v) if seed is None else seed
        for n in reversed(order):
            if n.bk is None or n.g is None: continue
            for p,g in zip(n.ps,n.bk(n.g)):
                p.g=g if p.g is None else p.g+g
def unb(g,sh):
    while g.ndim>len(sh): g=g.sum(0)
    for i,d in enumerate(sh):
        if d==1 and g.shape[i]!=1: g=g.sum(i,keepdims=True)
    return g
class Adam:
    def __init__(s,ps,b1=.9,b2=.98,eps=1e-9):
        s.ps=ps;s.m=[np.zeros_like(p.v) for p in ps];s.u=[np.zeros_like(p.v) for p in ps];s.b1,s.b2,s.eps=b1,b2,eps;s.n=0
    def step(s,lr):
        s.n+=1
        for i,p in enumerate(s.ps):
            if p.g is None: continue   # 本步没用到的参数不更新
            s.m[i]=s.b1*s.m[i]+(1-s.b1)*p.g; s.u[i]=s.b2*s.u[i]+(1-s.b2)*p.g*p.g
            p.v-=lr*(s.m[i]/(1-s.b1**s.n))/(np.sqrt(s.u[i]/(1-s.b2**s.n))+s.eps); p.g=None

import numpy as np
class Enc:
    def __init__(s,V,L,d=32,h=4,ff=64,N=2,seed=0,causal=False,n_cls=0,pre=False):
        r=np.random.default_rng(seed); s.V,s.L,s.d,s.h,s.ff,s.N,s.causal,s.pre=V,L,d,h,ff,N,causal,pre; s.ps=[]
        s.E=s.par(r.normal(0,d**-.5,(V,d))); s.P=s.par(r.normal(0,d**-.5,(L,d)))
        lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        s.layers=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,ff),ab=s.par(np.zeros(ff)),b=lin(ff,d),bb=s.par(np.zeros(d))) for _ in range(N)]
        if pre:   # GPT-2：残差层权重初始化乘 1/√(残差层数)
            for l in s.layers:
                l['o'].v*=1/np.sqrt(2*N); l['b'].v*=1/np.sqrt(2*N)
        s.head=None; s.rng=r
    def par(s,v): t=T(v); s.ps.append(t); return t
    def add_head(s,k):
        s.head=T(s.rng.normal(0,1/np.sqrt(s.d),(s.d,k))); s.ps.append(s.head); return s.head
    def attn(s,l,x):
        B,n,_=x.v.shape; h,dk=s.h,s.d//s.h
        sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3)
        Q,K,W=sp(x@l['q']),sp(x@l['k']),sp(x@l['v'])
        sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk))
        if s.causal:
            mask=np.tril(np.ones((n,n),bool)); o=sc; sc=T(np.where(mask,o.v,-1e9),(o,)); sc.bk=lambda g,mask=mask:(g*mask,)
        a=sc.softmax(); s.last=a.v
        return (a@W).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def encode(s,x):
        n=x.shape[1]; h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P
        for l in s.layers:
            if s.pre:
                h=h+s.attn(l,h.lnorm()); g=h.lnorm(); h=h+(((g@l['a'])+l['ab']).relu()@l['b'])+l['bb']
            else:
                h=(h+s.attn(l,h)).lnorm(); h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h.lnorm() if s.pre else h
    def lm(s,x): return s.encode(x)@s.E.t(1,0)
def ce(lg,y,w=None):
    """交叉熵。w 是每个位置的权重（如只算被遮的词）。返回 损失值, 概率节点, 反向种子"""
    p=lg.softmax(); idx=np.indices(y.shape); py=p.v[(*idx,y)]
    if w is None: w=np.ones(y.shape)
    W=w.sum()+1e-9; val=-(w*np.log(py+1e-12)).sum()/W
    sd=np.zeros(p.v.shape); sd[(*idx,y)]=-w/(py+1e-12)/W
    return val,p,sd
def noam(st,d,warm): return d**-.5*min(st**-.5,st*warm**-1.5)

"""迷你 CLIP：小图 + 文字，只学“哪句话配哪张图”。测从没配对过的组合和从没教过的任务。"""
import sys,json,numpy as np
SH=['横线','竖线','方框','十字','对角线','小点'];PO=['左上','右上','左下','右下']
TPL=[lambda s,p:[p,'的',s],lambda s,p:[s,'在',p],lambda s,p:['一张',p,'的',s,'的','图']]
vocab=['<p>','<e>']+SH+PO+['的','在','一张','图'];ID={w:i for i,w in enumerate(vocab)};V=len(vocab);LT=8;H=16;P=4;NP=16
HOLD={(0,3),(1,0),(2,2),(3,1),(4,3),(5,0)}   # (形状,位置) 训练时从不出现的组合
def draw(r,c,q):
    a=np.zeros((H,H));s=r.integers(4,7);oy=(q//2)*8;ox=(q%2)*8;x=ox+r.integers(0,8-s+1);y=oy+r.integers(0,8-s+1)
    if c==0:a[y+s//2,x:x+s]=1
    elif c==1:a[y:y+s,x+s//2]=1
    elif c==2:a[y,x:x+s]=1;a[y+s-1,x:x+s]=1;a[y:y+s,x]=1;a[y:y+s,x+s-1]=1
    elif c==3:a[y+s//2,x:x+s]=1;a[y:y+s,x+s//2]=1
    elif c==4:
        for i in range(s):a[y+i,x+i]=1
    else:a[y+s//2:y+s//2+2,x+s//2:x+s//2+2]=1
    return np.clip(a+r.normal(0,.1,a.shape),0,1)
def enc_text(ws):return [ID[w] for w in ws+['<e>']]+[0]*(LT-len(ws)-1)
def patches(X):
    B=len(X);return X.reshape(B,4,4,4,4).transpose(0,1,3,2,4).reshape(B,NP,16)
def l2(x):
    n=np.sqrt((x.v**2).sum(-1,keepdims=True)+1e-9);r=T(x.v/n,(x,));u=x.v/n
    r.bk=lambda g:((g-(g*u).sum(-1,keepdims=True)*u)/n,);return r
def scale(s,t):
    k=np.exp(t.v[0]);r=T(s.v*k,(s,t));r.bk=lambda g:(g*k,np.array([(g*s.v*k).sum()]));return r
class Blocks:
    def __init__(s,o,d,h,ff,N,r):
        lin=lambda a,b:o.par(r.normal(0,1/np.sqrt(a),(a,b)));s.d,s.h=d,h
        s.L=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,ff),ab=o.par(np.zeros(ff)),b=lin(ff,d),bb=o.par(np.zeros(d))) for _ in range(N)]
    def attn(s,l,x):
        B,n,_=x.v.shape;h,dk=s.h,s.d//s.h;sp=lambda t:t.rs(B,n,h,dk).t(0,2,1,3);Q,K,W=sp(x@l['q']),sp(x@l['k']),sp(x@l['v'])
        return ((Q@K.t(0,1,3,2)*(1/np.sqrt(dk))).softmax()@W).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def __call__(s,x):
        for l in s.L:
            x=x+s.attn(l,x.lnorm());g=x.lnorm();x=x+(((g@l['a'])+l['ab']).relu()@l['b'])+l['bb']
        return x.lnorm()
class CLIP:
    def __init__(s,seed=0,d=32,de=24):
        r=np.random.default_rng(seed);s.ps=[];s.d=d;lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        s.PE=lin(16,d);s.CL=s.par(r.normal(0,.1,(1,d)));s.PI=s.par(r.normal(0,.1,(NP+1,d)));s.VB=Blocks(s,d,4,64,2,r);s.VP=lin(d,de)
        s.TE=lin(V,d);s.PT=s.par(r.normal(0,.1,(LT,d)));s.TB=Blocks(s,d,4,64,2,r);s.TP=lin(d,de);s.logT=s.par(np.array([np.log(1/.07)]))
    def par(s,v):t=T(v);s.ps.append(t);return t
    def image(s,X):
        B=len(X);p=T(patches(X))@s.PE;cls=T(np.ones((B,1,1)))@s.CL;A=np.zeros((1,NP+1));A[0,0]=1
        x=T(np.tile(np.eye(NP+1)[:,1:],(B,1,1)))@p+T(np.tile(A.T,(B,1,1)))@cls+s.PI
        h=s.VB(x);pick=np.zeros((B,NP+1,1));pick[:,0,0]=1
        return l2((h.t(0,2,1)@T(pick)).rs(B,s.d)@s.VP)
    def text(s,ids):
        ids=np.array(ids);B=len(ids);x=T(np.eye(V)[ids])@s.TE+T(np.eye(LT))@s.PT;h=s.TB(x);pick=np.zeros((B,LT,1))
        pick[np.arange(B),(ids==1).argmax(1),0]=1
        return l2((h.t(0,2,1)@T(pick)).rs(B,s.d)@s.TP)
def sample(r,B):
    X=[];W=[]
    while len(X)<B:
        c,q=int(r.integers(6)),int(r.integers(4))
        if (c,q) in HOLD:continue
        X.append(draw(r,c,q));W.append(enc_text(TPL[r.integers(3)](SH[c],PO[q])))
    return np.stack(X),W
def step(m,o,r,lr,B=32):
    X,W=sample(r,B);I=m.image(X);Tx=m.text(W);S=I@Tx.t(1,0);L=scale(S,m.logT);v=L.v
    def sm(a,ax):
        e=np.exp(a-a.max(ax,keepdims=True));return e/e.sum(ax,keepdims=True)
    pr,pc=sm(v,1),sm(v,0);i=np.arange(B);loss=-.5*(np.log(pr[i,i]).mean()+np.log(pc[i,i]).mean())
    seed=.5*(pr+pc)/B;seed[i,i]-=1/B;L.backward(seed);o.step(lr);m.logT.v[0]=min(m.logT.v[0],np.log(100));return float(loss)
def emb(m,prompts):return m.text([enc_text(p) for p in prompts]).v
def test(m,seed=7,n=240):
    r=np.random.default_rng(seed);C=[(c,q) for c in range(6) for q in range(4)]
    res={}
    def acc(pairs,labels,txt,lab):
        X=np.stack([draw(r,c,q) for c,q in pairs]);I=m.image(X).v;return float((np.array(lab)[(I@txt.T).argmax(1)]==np.array(labels)).mean())
    seen=[C[i] for i in r.integers(0,24,n) if C[i] not in HOLD];new=[list(HOLD)[i] for i in r.integers(0,6,n)]
    ens=np.stack([emb(m,[t(SH[c],PO[q]) for c,q in C]) for t in TPL]).mean(0);ens/=np.linalg.norm(ens,axis=1,keepdims=True)
    one=emb(m,[TPL[0](SH[c],PO[q]) for c,q in C])
    res['seen']=acc(seen,[C.index(p) for p in seen],one,list(range(24)));res['new']=acc(new,[C.index(p) for p in new],one,list(range(24)));res['new_ens']=acc(new,[C.index(p) for p in new],ens,list(range(24)));res['seen_ens']=acc(seen,[C.index(p) for p in seen],ens,list(range(24)))
    allp=[C[i] for i in r.integers(0,24,n)]
    # 从没教过的任务：只问形状 / 只问位置（模板没出现过这种短语）
    sh=emb(m,[[s] for s in SH]);res['shape_only']=acc(allp,[c for c,q in allp],sh,list(range(6)))
    po=emb(m,[[p] for p in PO]);res['pos_only']=acc(allp,[q for c,q in allp],po,list(range(4)))
    # 平均成对的提示：对形状用含位置的模板做集成
    sh2=np.stack([np.mean([emb(m,[t(s,p)])[0] for t in TPL for p in PO],0) for s in SH]);sh2/=np.linalg.norm(sh2,axis=1,keepdims=True);res['shape_prompt']=acc(allp,[c for c,q in allp],sh2,list(range(6)))
    po2=np.stack([np.mean([emb(m,[t(s,p)])[0] for t in TPL for s in SH],0) for p in PO]);po2/=np.linalg.norm(po2,axis=1,keepdims=True);res['pos_prompt']=acc(allp,[q for c,q in allp],po2,list(range(4)))
    res['temp']=float(np.exp(m.logT.v[0]));return res
if __name__=='__main__':
    seed=int(sys.argv[1]);steps=int(sys.argv[2]);m=CLIP(seed);o=Adam(m.ps,b2=.98);r=np.random.default_rng(seed+1);h=[]
    for st in range(1,steps+1):
        l=step(m,o,r,min(st/100,1)*1.5e-3*(1-st/(steps+1)))
        if st%(steps//10)==0:h.append((st,round(l,4)));print(st,l,flush=True)
    out=dict(hist=h,params=sum(p.v.size for p in m.ps),test=test(m));print(out['test'],flush=True);json.dump(out,open(f'res{seed}.json','w'))
    if seed==0:np.savez('clip.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
