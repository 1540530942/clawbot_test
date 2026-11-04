#!/usr/bin/env python3
"""第 3 期配套：迷你 GPT。观察零样本能力随预训练增长，再比较微调方式。
只依赖 NumPy，运行 python3 03_mini_gpt.py（2 核约 10 分钟），结果写入当前目录。
"""
import sys,json
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
            s.m[i]=s.b1*s.m[i]+(1-s.b1)*p.g; s.u[i]=s.b2*s.u[i]+(1-s.b2)*p.g*p.g
            p.v-=lr*(s.m[i]/(1-s.b1**s.n))/(np.sqrt(s.u[i]/(1-s.b2**s.n))+s.eps); p.g=None
class Enc:
    def __init__(s,V,L,d=32,h=4,ff=64,N=2,seed=0,causal=False,n_cls=0):
        r=np.random.default_rng(seed); s.V,s.L,s.d,s.h,s.ff,s.N,s.causal=V,L,d,h,ff,N,causal; s.ps=[]
        s.E=s.par(r.normal(0,d**-.5,(V,d))); s.P=s.par(r.normal(0,d**-.5,(L,d)))
        lin=lambda a,b:s.par(r.normal(0,1/np.sqrt(a),(a,b)))
        s.layers=[dict(q=lin(d,d),k=lin(d,d),v=lin(d,d),o=lin(d,d),a=lin(d,ff),ab=s.par(np.zeros(ff)),b=lin(ff,d),bb=s.par(np.zeros(d))) for _ in range(N)]
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
            h=(h+s.attn(l,h)).lnorm(); h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h
    def lm(s,x): return s.encode(x)@s.E.t(1,0)
def ce(lg,y,w=None):
    """交叉熵。w 是每个位置的权重（如只算被遮的词）。返回 损失值, 概率节点, 反向种子"""
    p=lg.softmax(); idx=np.indices(y.shape); py=p.v[(*idx,y)]
    if w is None: w=np.ones(y.shape)
    W=w.sum()+1e-9; val=-(w*np.log(py+1e-12)).sum()/W
    sd=np.zeros(p.v.shape); sd[(*idx,y)]=-w/(py+1e-12)/W
    return val,p,sd
def noam(st,d,warm): return d**-.5*min(st**-.5,st*warm**-1.5)
POS=['精彩','感人','好看','有趣','出色','流畅'];NEG=['无聊','糟糕','难看','乏味','拖沓','平庸']
TOP=['电影','小说','游戏','音乐','餐厅'];ADV=['真的','确实','有点']
vocab=['<s>','<e>','<p>']+TOP+ADV+POS+NEG+['开头','但是','后来','。','总之','很','好','差']
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);L=13;PAD=ID['<p>']
def body(r,twist=None):
    """返回词列表和情感标签。twist=True 时是先抑后扬式转折句。"""
    if twist is None:twist=r.random()<.5
    s=r.random()<.5;pick=lambda b:(POS if b else NEG)[r.integers(6)]
    if twist:
        return ['<s>',TOP[r.integers(5)],'开头',pick(r.random()<.5),'但是','后来',pick(s),'。'],s
    return ['<s>',TOP[r.integers(5)],ADV[r.integers(3)],pick(s),'。'],s
def full(r):
    b,s=body(r)
    if r.random()<.7: b=b+['总之','很','好' if s else '差']
    return b+['<e>']
def pad(x,n=L): return x+['<p>']*(n-len(x))
def enc(ss,n=L): return np.array([[ID[w] for w in pad(s,n)] for s in ss])
def lm_batch(r,B):
    y=enc([full(r) for _ in range(B)]);return y[:,:-1],y[:,1:],(y[:,1:]!=PAD)*1.
def test_set(seed=11,n=1000):
    r=np.random.default_rng(seed);S=[];Y=[];TW=[]
    for i in range(n):
        tw=i%2==1;b,s=body(r,tw);S.append(b);Y.append(s);TW.append(tw)
    return S,np.array(Y),np.array(TW)
def zero(m,S,Y,TW):
    """零样本：不微调，接上“总之 很”，比较 好 和 差 的概率。"""
    pr=[]
    groups={}
    for i,b in enumerate(S): groups.setdefault(len(b),[]).append(i)
    pr=np.zeros(len(S),bool)
    for n,idx in groups.items():
        x=enc([S[i]+['总之','很'] for i in idx],n+2);lg=m.lm(x).v[:,-1];pr[idx]=lg[:,ID['好']]>lg[:,ID['差']]
    c=pr==Y;return float(c.mean()),float(c[~TW].mean()),float(c[TW].mean())
def pretrain(seed,steps=1500,every=100,tst=None):
    m=Enc(V,L,seed=seed,causal=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+50);h=[];z=[]
    for st in range(1,steps+1):
        x,y,w=lm_batch(r,64);val,p,sd=ce(m.lm(x),y,w);p.backward(sd);o.step(min(st/100,1)*2.5e-3*.5*(1+np.cos(np.pi*st/steps)))
        if st==1 or st%every==0:
            h.append((st,round(float(val),4)))
            if tst:z.append((st,)+zero(m,*tst))
    return m,h,z
def labeled(r,n):
    S=[];Y=[]
    for i in range(n):
        b,s=body(r,i%2==1);S.append(b+['<e>']);Y.append(int(s))
    return S,np.array(Y)
def clf_logits(m,H,x,last):
    h=m.encode(x);B,n,d=h.v.shape;pick=np.zeros((B,n,1));pick[np.arange(B),last,0]=1
    return (h.t(0,2,1)@T(pick)).rs(B,d)@H
def finetune(base,S,Y,tst,seed,aux,steps=120,lr=6e-4,lam=.5):
    n=Enc(V,L,seed=seed+500,causal=True)
    if base is not None:
        for a,b in zip(n.ps,base.ps):a.v=b.v.copy()
    n.rng=np.random.default_rng(seed+900);H=n.add_head(2);o=Adam(n.ps,b2=.999)
    x=enc(S,L);last=np.array([len(s)-1 for s in S])
    for st in range(1,steps+1):
        c=clf_logits(n,H,x,last);v,p,sd=ce(c,Y);p.backward(sd)
        if aux:
            xx=x[:,:-1];yy=x[:,1:];w=(yy!=PAD)*1.;v2,p2,sd2=ce(n.lm(xx),yy,w);p2.backward(sd2*lam)
        o.step(lr*(1-st/(steps+1)))
    TS,Yt,TW=tst;xt=enc([s+['<e>'] for s in TS],L);lt=np.array([len(s) for s in TS])
    c=clf_logits(n,H,xt,lt).v.argmax(-1)==Yt.astype(int);return float(c.mean()),float(c[~TW].mean()),float(c[TW].mean())
def gradcheck():
    r=np.random.default_rng(0);m=Enc(V,L,d=8,h=2,ff=8,N=1,seed=0,causal=True);x,y,w=lm_batch(r,4);v,p,sd=ce(m.lm(x),y,w);p.backward(sd);worst=0
    for P in m.ps[:5]:
        i=tuple(r.integers(0,s) for s in P.v.shape);g=P.g[i];o=P.v[i];e=1e-5
        P.v[i]=o+e;a=ce(m.lm(x),y,w)[0];P.v[i]=o-e;b=ce(m.lm(x),y,w)[0];P.v[i]=o;worst=max(worst,abs((a-b)/(2*e)-g))
    return worst
if __name__=='__main__':
    print('grad',gradcheck(),'V',V,flush=True);tst=test_set();out=dict(z=[],h=[],ft=[])
    for seed in range(3):
        m,h,z=pretrain(seed,tst=tst);out['z'].append(z);out['h'].append(h);print(seed,'loss',h[-1],'zero',z[0],z[-1],flush=True)
        if seed==0:np.savez('gpt.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)});json.dump(vocab,open('vocab.json','w'),ensure_ascii=False)
        for nl in (10,30,100):
            rr=np.random.default_rng(seed*7+nl);S,Y=labeled(rr,nl)
            res=dict(n=nl,seed=seed,full=finetune(m,S,Y,tst,seed,True),noaux=finetune(m,S,Y,tst,seed,False),none=finetune(None,S,Y,tst,seed,False));out['ft'].append(res);print(res,flush=True)
        json.dump(out,open('res.json','w'))
