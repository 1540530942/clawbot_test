#!/usr/bin/env python3
# 第 6 期迷你上下文学习：python3 06_mini_incontext.py 0 2500
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

"""迷你 GPT-3：语料里每段文字都用一套随机“暗号表”。测试时给全新的暗号表和 K 个示例，不做梯度更新，看能否当场学会。"""
import sys,json,numpy as np
LET=list('ABCDEFGH');DIG=[str(i) for i in range(8)]
vocab=['<s>','<e>','<p>']+LET+DIG+['=',';','?']
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);L=29;PAD=ID['<p>']
def doc(r,K=None):
    """一段：K 个示例 + 问题。暗号表（字母到数字的随机映射）每段都不同。"""
    m=r.permutation(8);K=int(r.integers(0,7)) if K is None else K
    pos=r.permutation(8);demos=[];nx=0
    for i in range(K):      # 一半概率重复前面出现过的字母，这样“回头查”才是有用的技能
        if demos and r.random()<.5:demos.append(int(demos[r.integers(len(demos))]))
        else:demos.append(int(pos[nx]));nx+=1
    q=int(demos[r.integers(K)]) if K>0 and r.random()<.75 else int(pos[nx%8])
    w=['<s>']
    for i in demos:w+=[LET[i],'=',DIG[m[i]],';']
    w+=[LET[q],'=',DIG[m[q]],'<e>'];return w
def enc(ss,n=L):return np.array([[ID[w] for w in s+['<p>']*(n-len(s))] for s in ss])
def batch(r,B):
    y=enc([doc(r) for _ in range(B)]);x,t=y[:,:-1],y[:,1:];w=(t!=PAD)*0.1
    # 重点算答案位（=后面的数字），其他位权重小
    eq=x==ID['='];w[eq]=1.;return x,t,w
def test(m,n=600,seed=3):
    r=np.random.default_rng(seed);res={}
    for K in range(0,7):
        ok=seen=tot=0;oks=0
        P=[]
        for _ in range(n):
            mp=r.permutation(8);pos=r.permutation(8);demos=[int(pos[i]) for i in range(K)]
            q=int(demos[r.integers(K)]) if K>0 else int(pos[0])
            w=['<s>']
            for i in demos:w+=[LET[i],'=',DIG[mp[i]],';']
            P.append((w+[LET[q],'='],ID[DIG[mp[q]]]))
        X=enc([p[0] for p in P],len(P[0][0]));lg=m.lm(X).v[:,-1];res[K]=float(np.mean(lg.argmax(-1)==np.array([p[1] for p in P])))
    return res
SIZES=[(16,1,32),(32,2,64),(48,3,96)]
if __name__=='__main__':
    seed=int(sys.argv[1]);steps=int(sys.argv[2]);out=[]
    for k,(d,N,ff) in enumerate(SIZES):
        m=Enc(V,L,d=d,h=4,ff=ff,N=N,seed=seed,causal=True,pre=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+7);h=[]
        for st in range(1,steps+1):
            x,t,w=batch(r,64);v,p,sd=ce(m.lm(x),t,w);p.backward(sd);o.step(min(st/200,1)*3e-3*.5*(1+np.cos(np.pi*st/steps)))
            if st%(steps//5)==0:h.append((st,test(m,200)))
        res=dict(d=d,N=N,params=sum(p.v.size for p in m.ps),hist=h,final=test(m));out.append(res);print(seed,k,res['params'],res['final'],flush=True)
        if seed==0 and k==2:np.savez('gpt3.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
        json.dump(out,open(f'res{seed}.json','w'))
