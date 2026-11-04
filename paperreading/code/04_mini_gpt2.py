#!/usr/bin/env python3
"""第 4 期配套：四种容量的迷你语言模型，测零样本任务。
只依赖 NumPy：python3 04_mini_gpt2.py 0 2500（种子、步数，2 核约 30 分钟）。
"""
import sys,json,os
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
NAMES=['小明','小红','小刚','小丽','小华','小军','小芳','小强','小燕','小伟','小静','小勇','小霞','小波','小敏','小杰','小雪','小龙','小玲','小峰','小梅','小辉','小倩','小涛']
CITY=['北京','上海','广州','成都','西安','杭州','武汉','南京']
NOUN=['苹果','香蕉','火车','飞机','月亮','太阳','书本','钢笔','小河','大山']
DIG=[str(i) for i in range(10)]
vocab=['<s>','<e>','<p>']+NAMES+CITY+NOUN+DIG+['住','在','。','问','哪','答','总之','接着','+','=']
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);L=9;PAD=ID['<p>']
HOME={n:CITY[(i*5+3)%8] for i,n in enumerate(NAMES)}
QA_TRAIN=NAMES[:16];QA_TEST=NAMES[16:]
PAIRS=[(a,b) for a in range(10) for b in range(10)]
_r=np.random.default_rng(1234);_p=_r.permutation(100);ADD_TRAIN=[PAIRS[i] for i in _p[:70]];ADD_TEST=[PAIRS[i] for i in _p[70:]]
def doc(r):
    u=r.random()
    if u<.3:
        n=NAMES[r.integers(24)];return ['<s>',n,'住','在',HOME[n],'。','<e>']
    if u<.5:
        n=QA_TRAIN[r.integers(16)];return ['<s>','问',n,'住','哪','答',HOME[n],'<e>']
    if u<.75:
        w=[NOUN[i] for i in r.integers(0,10,4)]
        if r.random()<.5:return ['<s>']+w+['。','总之',w[0],'<e>']
        return ['<s>']+w+['。','接着',NOUN[r.integers(10)],'<e>']
    a,b=ADD_TRAIN[r.integers(70)];return ['<s>',str(a),'+',str(b),'=',str((a+b)%10),'<e>']
def enc(ss,n=L):
    return np.array([[ID[w] for w in s+['<p>']*(n-len(s))] for s in ss])
def batch(r,B):
    y=enc([doc(r) for _ in range(B)]);return y[:,:-1],y[:,1:],(y[:,1:]!=PAD)*1.
def nextdist(m,pre):
    x=enc(pre,len(pre[0]));return m.lm(x).v[:,-1]
def tasks(m):
    r=np.random.default_rng(99);o={}
    Q=lambda ns:float(np.mean(nextdist(m,[['<s>','问',n,'住','哪','答'] for n in ns]).argmax(-1)==np.array([ID[HOME[n]] for n in ns])))
    o['qa_seen']=Q(QA_TRAIN);o['qa_new']=Q(QA_TEST)
    F=[(n,) for n in NAMES];o['fact']=float(np.mean(nextdist(m,[['<s>',n,'住','在'] for n in NAMES]).argmax(-1)==np.array([ID[HOME[n]] for n in NAMES])))
    W=[[NOUN[i] for i in r.integers(0,10,4)] for _ in range(400)];Y=np.array([ID[w[0]] for w in W])
    o['hint']=float(np.mean(nextdist(m,[['<s>']+w+['。','总之'] for w in W]).argmax(-1)==Y))
    nid=[ID[w] for w in NOUN];lg=nextdist(m,[['<s>']+w+['。','接着'] for w in W])[:,nid];o['nohint']=float(np.mean(np.array(nid)[lg.argmax(-1)]==Y))
    A=lambda P:float(np.mean(nextdist(m,[['<s>',str(a),'+',str(b),'='] for a,b in P]).argmax(-1)==np.array([ID[str((a+b)%10)] for a,b in P])))
    o['add_seen']=A(ADD_TRAIN);o['add_new']=A(ADD_TEST)
    x,y,w=batch(np.random.default_rng(5),800);v,_,_=ce(m.lm(x),y,w);o['ppl']=float(np.exp(v));return o
SIZES=[(16,1,32),(32,2,64),(48,3,96),(64,4,128)]
if __name__=='__main__':
    seed=int(sys.argv[1]);steps=int(sys.argv[2]) if len(sys.argv)>2 else 3000;out=[]
    for k,(d,N,ff) in enumerate(SIZES):
        m=Enc(V,L,d=d,h=4,ff=ff,N=N,seed=seed,causal=True,pre=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+77);nparam=sum(p.v.size for p in m.ps);cur=[]
        for st in range(1,steps+1):
            x,y,w=batch(r,64);v,p,sd=ce(m.lm(x),y,w);p.backward(sd);o.step(min(st/200,1)*3e-3*.5*(1+np.cos(np.pi*st/steps)))
            if st in (steps//4,steps//2,steps*3//4,steps):cur.append((st,tasks(m)))
        res=dict(size=k,d=d,N=N,params=nparam,hist=cur,final=cur[-1][1]);out.append(res);print(seed,k,nparam,cur[-1][1],flush=True)
        if seed==0 and k in (0,2):np.savez(f'm{k}.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
        json.dump(out,open(f'res{seed}.json','w'))
