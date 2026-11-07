#!/usr/bin/env python3
# 第 34 期迷你 Flamingo（8000 步版）
# 单文件版：下面依次是极小的自动微分、迷你 Transformer 和实验本身，只依赖 NumPy。
import gc
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
        if s.n%20==0: gc.collect()   # 计算图可能有引用环，定期回收防止内存涨
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

"""第 34 期玩具实验：迷你 Flamingo。视觉编码器（先训分类再冻结）+ 冻结的迷你语言模型 + 只训练的感知重采样器和门控交叉注意力。
“网页”语料里每篇文档连续几张图都用同一种说法（数字名、奇偶、大小、相邻、形状），模型要从前面的例子判断这篇在用哪种说法。
比较：门控和不门控、冻结和不冻结、有无交错语料、只看前一张图和看所有图。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
G=["..XXXX..,.X....X.,X......X,X......X,X......X,X......X,.X....X.,..XXXX..","...XX...,..XXX...,...XX...,...XX...,...XX...,...XX...,...XX...,.XXXXXX.",".XXXXX..,X.....X.,......X.,.....X..,...XX...,..X.....,.X......,XXXXXXXX",".XXXXX..,......X.,......X.,..XXXX..,......X.,......X.,......X.,.XXXXX..",".....X..,....XX..,...X.X..,..X..X..,.X...X..,XXXXXXXX,.....X..,.....X..","XXXXXXX.,X.......,X.......,XXXXXX..,......X.,......X.,X.....X.,.XXXXX..","..XXXX..,.X......,X.......,XXXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..","XXXXXXXX,......X.,.....X..,....X...,...X....,...X....,..X.....,..X.....",".XXXXX..,X.....X.,X.....X.,.XXXXX..,X.....X.,X.....X.,X.....X.,.XXXXX..",".XXXXX..,X.....X.,X.....X.,.XXXXXX.,......X.,......X.,.....X..,.XXXX..."]
BM=np.array([[[c=='X' for c in row] for row in g.split(',')] for g in G],float)
NAME=['zero','one','two','three','four','five','six','seven','eight','nine']
words=['<p>','<image>','<eoc>','the','a','is','and','it','word','number','this','that','odd','even','big','small','round','straight','output']+NAME
ID={w:i for i,w in enumerate(words)};PAD,IMG,EOC=0,1,2;V=len(words);L=44
TASK={'name':lambda d:NAME[d],'parity':lambda d:'odd' if d%2 else 'even','size':lambda d:'big' if d>=5 else 'small','next':lambda d:NAME[(d+1)%10],'shape':lambda d:'round' if d in(0,3,6,8,9) else 'straight'}
TK=list(TASK)
def image(r,d):
    x=np.roll(BM[d],int(r.integers(-1,2)),1)+r.normal(0,.2,(8,8));return x.reshape(-1)
def doc(r,task,n,shots=None):
    """n 张图的交错文档：<image> 输出 <eoc>"""
    toks=[];imgs=[];ds=[]
    for i in range(n):
        d=int(r.integers(0,10));toks+=[IMG,ID[TASK[task](d)],EOC];imgs.append(image(r,d));ds.append(d)
    return toks,imgs,ds
def pair(r):
    d=int(r.integers(0,10));return [IMG,ID[NAME[d]],EOC],[image(r,d)],[d]
def text_doc(r):
    """纯文本语料：语言模型的预训练；有简单的词序规律和复述"""
    a,b=[NAME[int(i)] for i in r.integers(0,10,2)];s=[('the','number',a,'is','big'),('the','word',a,'and','the','word',b),('this','is',a,'and','this','is',a),('a','number','is','odd','or' if False else 'and','even'),('it','is',a,'it','is',a),('small','and','big'),('odd','and','even'),('round','and','straight')][int(r.integers(0,8))];return [ID[w] for w in s]+[EOC]
sg=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
def tanh(x):
    p=np.tanh(x.v);r=T(p,(x,));r.bk=lambda g:(g*(1-p*p),);return r
class VE:
    """冻结的视觉编码器：线性切 4 个图块 → 特征，用类别预训练（代替对比学习）"""
    def __init__(s,seed,d=32):
        r=np.random.default_rng(seed);s.W=r.normal(0,.2,(16,d));s.C=r.normal(0,.1,(d*4,10));s.b=np.zeros(d);s.d=d
    def patch(s,x):
        n=len(x);a=x.reshape(n,8,8);return np.stack([a[:,:4,:4],a[:,:4,4:],a[:,4:,:4],a[:,4:,4:]],1).reshape(n,4,16)
    def fw(s,x):return np.maximum(s.patch(x)@s.W+s.b,0)
    def train(s,r,steps):
        M=[0,0,0];U=[0,0,0]
        for st in range(1,steps+1):
            y=r.integers(0,10,64);x=np.stack([image(r,int(d)) for d in y]);p=s.patch(x);h=np.maximum(p@s.W+s.b,0);f=h.reshape(64,-1);z=f@s.C;q=z-z.max(1,keepdims=True);e=np.exp(q);e/=e.sum(1,keepdims=True);e[np.arange(64),y]-=1;dh=(e@s.C.T).reshape(64,4,s.d)*(h>0);G=[(f.T@e)/64,np.einsum('npk,npd->kd',p,dh)/64,dh.mean((0,1))]
            for i,(a,g) in enumerate(zip((s.C,s.W,s.b),G)):
                M[i]=.9*M[i]+.1*g;U[i]=.999*U[i]+.001*g*g;a-=2e-3*(M[i]/(1-.9**st))/(np.sqrt(U[i]/(1-.999**st))+1e-8)
        y=r.integers(0,10,500);x=np.stack([image(r,int(d)) for d in y]);return float((((s.fw(x).reshape(500,-1))@s.C).argmax(1)==y).mean())
class Fl(Enc):
    def setup(s,seed,gate=True,per_image=True,nv=4):
        r=np.random.default_rng(seed+30);s.gate,s.per=gate,per_image;d=s.d;lin=lambda a,b,sc=1.:s.par(r.normal(0,sc/np.sqrt(a),(a,b)))
        s.lmps=list(s.ps);s.ps=[]
        s.Lat=s.par(r.normal(0,.1,(nv,d)));s.rk=lin(32,d);s.rv=lin(32,d);s.rq=lin(d,d);s.ro=lin(d,d);s.nv=nv
        s.x=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,2*d),ab=s.par(np.zeros(2*d)),b=lin(2*d,d),bb=s.par(np.zeros(d)),al=s.par(np.zeros((1,1))),bl=s.par(np.zeros((1,1)))) for _ in s.layers]
        s.newps=list(s.ps);s.ps=s.lmps+s.newps
    def resample(s,F):
        """感知重采样器：可学习的潜变量做查询，对图块特征做交叉注意力"""
        B=F.shape[0];lat=T(np.ones((B,1,1)))*s.Lat;K=T(F)@s.rk;Vv=T(F)@s.rv;Q=lat@s.rq;a=(Q@K.t(0,2,1)*(1/np.sqrt(s.d))).softmax();return (lat+(a@Vv)@s.ro).lnorm()
    def xattn(s,l,h,Vt,mask):
        B,n,d=h.v.shape;m=Vt.v.shape[1];hh=s.h;dk=d//hh
        sp=lambda t,k:t.rs(B,k,hh,dk).t(0,2,1,3);Q=sp(h@l['q'],n);K=sp(Vt@l['k'],m);Vv=sp(Vt@l['v'],m);sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk));mk=mask[:,None,:,:];o=sc;sc=T(np.where(mk,o.v,-1e9),(o,));sc.bk=lambda g,mk=mk:(g*mk,);a=sc.softmax();y=(a@Vv).t(0,2,1,3).rs(B,n,hh*dk)@l['o']
        if s.gate:
            ga=tanh(l['al']);gb=tanh(l['bl']);h=h+y*ga;h=h+(((h@l['a'])+l['ab']).relu()@l['b']+l['bb'])*gb
        else:
            h=h+y;h=h+(((h@l['a'])+l['ab']).relu()@l['b']+l['bb'])
        return h
    def fwd(s,x,F,idx):
        """x:(B,n) 词；F:(B,NI,4,32) 图块特征；idx:(B,n) 每个词对应的最后一张图编号（-1 表示没有）"""
        B,n=x.shape;h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.P;NI=F.shape[1];Fl_=F.reshape(B*NI,4,32);R=s.resample(Fl_).rs(B,NI*s.nv,s.d)
        mask=np.zeros((B,n,NI*s.nv),bool)
        for i in range(NI):
            sel=(idx==i) if s.per else (idx>=i);mask[:,:,i*s.nv:(i+1)*s.nv]=sel[:,:,None]
        mask[(idx<0)]=False;nm=~mask.any(-1);mask[nm,0]=True  # 没有图时看到一个（门控会学着不靠它）
        for l,xl in zip(s.layers,s.x):
            h=s.xattn(xl,h,R,mask);h=(h+s.attn(l,h)).lnorm();h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h@s.E.t(1,0)
def collate(batch,ve,maxi=None):
    B=len(batch);n=max(len(t) for t,_,_ in batch);x=np.zeros((B,n),int);NI=max(len(i) for _,i,_ in batch);F=np.zeros((B,NI,4,32));idx=-np.ones((B,n),int)
    for b,(t,im,_) in enumerate(batch):
        x[b,:len(t)]=t;k=-1
        for j,w in enumerate(t):
            if w==IMG:k+=1
            idx[b,j]=k
        if im:F[b,:len(im)]=ve.fw(np.stack(im))
    return x,F,idx
def make(seed,**kw):
    m=Fl(V,L,d=32,h=4,ff=64,N=2,seed=seed,causal=True);return m
def train_lm(m,r,steps):
    o=Adam(m.ps)
    for st in range(1,steps+1):
        S=[text_doc(r) for _ in range(32)];n=max(map(len,S));x=np.zeros((32,n),int)
        for i,s_ in enumerate(S):x[i,:len(s_)]=s_
        X,y=x[:,:-1],x[:,1:];v,p,g=ce(m.lm(X),y,(y!=PAD)*1.);p.backward(g);o.step(2e-3*(1 if st<.8*steps else .3))
def lm_nll(m,seed):
    r=np.random.default_rng(seed);S=[text_doc(r) for _ in range(200)];n=max(map(len,S));x=np.zeros((200,n),int)
    for i,s_ in enumerate(S):x[i,:len(s_)]=s_
    return float(ce(m.lm(x[:,:-1]),x[:,1:],(x[:,1:]!=PAD)*1.)[0])
def train_fl(m,ve,r,steps,mode='mix',freeze=True,tasks=TK,lr=1e-3):
    tr=m.newps if freeze else m.ps;o=Adam(tr);h=[]
    for st in range(1,steps+1):
        batch=[]
        for _ in range(16):
            if mode=='pairs' or (mode=='mix' and r.random()<.2):batch.append(pair(r))
            else:batch.append(doc(r,tasks[int(r.integers(0,len(tasks)))],int(r.integers(1,6))))
        x,F,idx=collate(batch,ve);lg=m.fwd(x[:,:-1],F,idx[:,:-1]);y=x[:,1:];w=((y!=PAD)&(y!=IMG))*1.;v,p,g=ce(lg,y,w);p.backward(g)
        if freeze:
            for q in m.lmps:q.g=None
        o.step(lr*min(1,st/100)*(.3 if st>.8*steps else 1))
        if st%max(1,steps//8)==0:h.append((st,round(float(v),3)))
    return h
def score(m,ve,seed,task,shots,n=120):
    """shots 个示例后，看查询图的输出（只比最可能的词）。0 示例就只给图"""
    r=np.random.default_rng(seed);ok=0;batch=[];ans=[]
    for _ in range(n):
        t,im,ds=doc(r,task,shots+1);q=t[:-2];batch.append((q,im,ds));ans.append(t[-2])
    x,F,idx=collate(batch,ve);lg=m.fwd(x,F,idx).v[:,-1];lg[:,[PAD,IMG,EOC]]=-1e9;return float((lg.argmax(1)==np.array(ans)).mean())
def evaluate(m,ve,seed):
    return {t:{s_:score(m,ve,seed+i,t,s_) for s_ in (0,1,2,4,8)} for i,t in enumerate(TK)}
def clone_lm(m,**kw):
    n=make(0)
    for p,q in zip(n.ps,m.ps):p.v=q.v.copy()
    n.setup(kw.pop('seed',0),**kw);return n
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 2000;Sl=int(sys.argv[3]) if len(sys.argv)>3 else 1500;out={};t0=time.time();r=np.random.default_rng(sd+1)
    if S<=0:
        ve=VE(0);m=make(0);m.setup(0);batch=[doc(r,'name',2),doc(r,'parity',3)];x,F,idx=collate(batch,ve);
        def L_():return ce(m.fwd(x[:,:-1],F,idx[:,:-1]),x[:,1:],(x[:,1:]!=PAD)*1.)
        for l in m.x:l['al'].v[:]=.5;l['bl'].v[:]=.3
        v,p,g=L_();p.backward(g)
        for nm,q in (('Lat',m.Lat),('rk',m.rk),('xq',m.x[0]['q']),('al',m.x[1]['al']),('xa',m.x[1]['a'])):
            Gq=q.g.copy();i=np.unravel_index(int(np.argmax(abs(Gq))),Gq.shape);e=1e-6;q.v[i]+=e;a=L_()[0];q.v[i]-=2*e;b=L_()[0];q.v[i]+=e;print(nm,Gq[i],(a-b)/2/e)
        sys.exit()
    ve=VE(sd);out['ve']=ve.train(r,800);print(sd,'视觉',out['ve'],flush=True);base=make(sd);train_lm(base,r,Sl);n0=lm_nll(base,5);out['lm_nll']=n0;print(sd,'语言模型 NLL',n0,round(time.time()-t0),flush=True)
    arms=[('full',dict(),dict()),('pairs',dict(),dict(mode='pairs')),('nogate',dict(gate=False),dict()),('allimg',dict(per_image=False),dict()),('unfrozen',dict(),dict(freeze=False))]
    for k,mk,tk in arms:
        m=clone_lm(base,seed=sd,**mk);h=train_fl(m,ve,np.random.default_rng(sd+2),S,**tk);res=evaluate(m,ve,900+sd);a={s_:float(np.mean([res[t][s_] for t in TK])) for s_ in (0,1,2,4,8)};out[k]=dict(ev=res,avg=a,hist=h,lm_nll=lm_nll(m,5),al=[float(np.tanh(x['al'].v[0,0])) for x in m.x]);print(sd,k,a,'LM NLL',out[k]['lm_nll'],out[k]['al'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0 and k=='full':
            A={f'p{i}':p.v for i,p in enumerate(m.ps)};np.savez('m.npz',**A,veW=ve.W,veb=ve.b)
    # 保留任务：训练时不见 shape，测它从示例里学新说法（论文讨论的“任务定位”问题）
    m=clone_lm(base,seed=sd);train_fl(m,ve,np.random.default_rng(sd+2),S,tasks=[t for t in TK if t!='shape']);out['heldout']={s_:score(m,ve,1900+sd,'shape',s_) for s_ in (0,1,2,4,8)};out['heldout_seen']={s_:float(np.mean([score(m,ve,1900+sd,t,s_) for t in TK if t!='shape'])) for s_ in (0,1,2,4,8)};print(sd,'保留任务',out['heldout'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
