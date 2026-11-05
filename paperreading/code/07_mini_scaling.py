#!/usr/bin/env python3
# 第 07 期玩具实验：字符模型上的规模律（语料换成任意英文文本即可）
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

"""真实英文文本（arXiv 论文正文）上训练字符级小模型，拟合 L(N)、L(D)。"""
import sys,json,glob,re,numpy as np
def load():
    t=''
    for f in ['01','02','05','06','07']:
        s=open(f'../../papers/{f}/paper.txt',encoding='utf-8').read()
        s=s[:s.find('\nReferences')] if '\nReferences' in s else s
        s='\n'.join(l for l in s.split('\n') if len(l)>80 and not l.strip().startswith('|') and '$' not in l)
        t+=s+'\n'
    t=re.sub(r'[^\x20-\x7e\n]','',t);t=re.sub(r'\[[0-9, ]+\]','',t);return t
TXT=load();CH=sorted(set(TXT));V=len(CH);ID={c:i for i,c in enumerate(CH)};A=np.array([ID[c] for c in TXT])
cut=int(len(A)*.9);TR,TE=A[:cut],A[cut:];CTX=48
def get(src,r,B):
    i=r.integers(0,len(src)-CTX-1,B);x=np.stack([src[j:j+CTX] for j in i]);y=np.stack([src[j+1:j+CTX+1] for j in i]);return x,y
def nonemb(d,N,ff,h=4):
    return N*(4*(d*d+d)+(d*ff+ff)+(ff*d+d))
def tloss(m,n=100):
    r=np.random.default_rng(1);tot=0
    for _ in range(4):
        x,y=get(TE,r,n//4);tot+=ce(m.lm(x),y)[0]
    return tot/4
def train(d,N,ff,D=None,steps=2000,seed=0,every=100):
    m=Enc(V,CTX,d=d,h=4,ff=ff,N=N,seed=seed,causal=True,pre=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+3);src=TR if D is None else TR[:D]
    h=[];best=9e9
    for st in range(1,steps+1):
        x,y=get(src,r,32);v,p,sd=ce(m.lm(x),y);p.backward(sd);o.step(min(st/100,1)*3e-3*.5*(1+np.cos(np.pi*st/steps)))
        if st%every==0:
            l=tloss(m);h.append((st,round(float(v),4),round(float(l),4)));best=min(best,l)
    return m,h,float(best)
SZ=[(8,1,16),(16,1,32),(16,2,32),(32,2,64),(32,4,64),(48,3,96)]
if __name__=='__main__':
    print('语料字符',len(A),'字表',V,flush=True);seed=int(sys.argv[1])+2;steps=int(sys.argv[2])
    if seed==3:sys.exit()   # 种子 1 已有完整结果，只补种子 2
    out=dict(N=[],D=[],V=V,chars=len(A))
    for d,N,ff in SZ:
        m,h,b=train(d,N,ff,steps=steps,seed=seed)
        if seed==2 and (d,N) in ((16,1),(48,3)):np.savez(f'm{d}.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)});json.dump(CH,open('ch.json','w'))
        out['N'].append(dict(d=d,layers=N,params=nonemb(d,N,ff),best=b,h=h));print('N',d,N,nonemb(d,N,ff),b,flush=True);json.dump(out,open(f'res{seed}.json','w'))
    for D in (2000,8000,32000,128000,len(TR)):
        m,h,b=train(32,2,64,D=D,steps=steps,seed=seed);out['D'].append(dict(D=D,best=b,h=h,final=h[-1]));print('D',D,b,h[-1],flush=True);json.dump(out,open(f'res{seed}.json','w'))
