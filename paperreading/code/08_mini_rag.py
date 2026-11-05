#!/usr/bin/env python3
# 第 8 期迷你 RAG：python3 08_mini_rag.py 0 2500
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

"""迷你 RAG：闭卷模型（背住事实）对比检索加生成（读资料再答）。测热替换知识库和没见过的人。"""
import sys,json,math,numpy as np
NAMES=[a+b for a in '小大老阿' for b in '明红刚丽华军芳强燕伟静勇']   # 48 个人
CITY=['北京','上海','广州','成都','西安','杭州','武汉','南京','重庆','天津','苏州','青岛']
TRAIN=NAMES[:36];NEW=NAMES[36:]
vocab=['<s>','<e>','<p>']+NAMES+CITY+['问','住','哪','答','资料','在','。']
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);L=22;PAD=ID['<p>']
def kb(seed):
    r=np.random.default_rng(seed);return {n:CITY[r.integers(12)] for n in NAMES}
def docs(K):return [[n,'住','在',K[n],'。'] for n in NAMES]
def bm25(q,D,k1=1.5,b=.75):
    N=len(D);avg=sum(len(d) for d in D)/N;df={}
    for d in D:
        for w in set(d):df[w]=df.get(w,0)+1
    s=[]
    for d in D:
        t=0
        for w in set(q):
            if w in d:
                f=d.count(w);idf=math.log(1+(N-df[w]+.5)/(df[w]+.5));t+=idf*f*(k1+1)/(f+k1*(1-b+b*len(d)/avg))
        s.append(t)
    return np.array(s)
def question(n):return ['<s>','问',n,'住','哪','答']
def rag_input(n,ds):
    w=['<s>','资料']
    for d in ds:w+=d
    return w+['问',n,'住','哪','答']
def enc(ss):
    m=max(len(s) for s in ss);return np.array([[ID[w] for w in s+['<p>']*(m-len(s))] for s in ss])
def rag_batch(r,B):
    """每条样本都用随机知识，所以背不住，只能读资料。2 条资料里有 1 条正确（和 k=2 检索一样）。"""
    S=[];Y=[]
    for _ in range(B):
        i=r.integers(36);o=r.integers(36)
        while o==i:o=r.integers(36)
        a,b=TRAIN[i],TRAIN[o];ca,cb=CITY[r.integers(12)],CITY[r.integers(12)]
        ds=[[a,'住','在',ca,'。'],[b,'住','在',cb,'。']]
        if r.random()<.5:ds=ds[::-1]
        S.append(rag_input(a,ds)+[ca]);Y.append(ca)
    return S
def train_rag(seed,steps,d=48,N=2,ff=96):
    m=Enc(V,L,d=d,h=4,ff=ff,N=N,seed=seed,causal=True,pre=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+5);h=[]
    for st in range(1,steps+1):
        S=rag_batch(r,48);y=enc(S);x,t=y[:,:-1],y[:,1:];w=np.zeros(t.shape);w[:,-1]=1;v,p,sd=ce(m.lm(x),t,w);p.backward(sd);o.step(min(st/100,1)*3e-3*(1-st/(steps+1)))
        if st%100==0:h.append((st,round(float(v),4)))
    return m,h
def train_closed(seed,steps,K,d=48,N=2,ff=96):
    """闭卷：只在知识库 K 上背问答。"""
    m=Enc(V,L,d=d,h=4,ff=ff,N=N,seed=seed+100,causal=True,pre=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+6)
    for st in range(1,steps+1):
        S=[question(n)+[K[n]] for n in [TRAIN[i] for i in r.integers(0,36,48)]];y=enc(S);x,t=y[:,:-1],y[:,1:];w=np.zeros(t.shape);w[:,-1]=1;v,p,sd=ce(m.lm(x),t,w);p.backward(sd);o.step(min(st/100,1)*3e-3*(1-st/(steps+1)))
    return m
def ans(m,S):
    x=enc(S);lg=m.lm(x).v[:,-1];idx=[ID[c] for c in CITY];return np.array(CITY)[lg[:,idx].argmax(-1)]
def eval_closed(m,K,names):return float(np.mean(ans(m,[question(n) for n in names])==np.array([K[n] for n in names])))
def eval_rag(m,K_idx,K_truth,names,k=2):
    D=docs(K_idx);S=[];hit=0
    for n in names:
        s=bm25(question(n),D);o=s.argsort()[::-1][:k];ds=[D[i] for i in o];hit+=any(d[0]==n for d in ds);S.append(rag_input(n,ds))
    return float(np.mean(ans(m,S)==np.array([K_truth[n] for n in names]))),hit/len(names)
if __name__=='__main__':
    seed=int(sys.argv[1]);steps=int(sys.argv[2]);KA,KB=kb(1),kb(2);out={}
    M=train_rag(seed,steps);m=M[0];c=train_closed(seed,steps,KA)
    print(sum(p.v.size for p in m.ps),M[1][-1],flush=True)
    out=dict(params=sum(p.v.size for p in m.ps),hist=M[1],closed_A_on_A=eval_closed(c,KA,TRAIN),closed_A_on_B=eval_closed(c,KB,TRAIN),closed_new=eval_closed(c,KA,NEW),
      rag_AA=eval_rag(m,KA,KA,TRAIN),rag_BB=eval_rag(m,KB,KB,TRAIN),rag_mis_AB=eval_rag(m,KA,KB,TRAIN),rag_mis_BA=eval_rag(m,KB,KA,TRAIN),rag_new=eval_rag(m,KA,KA,NEW))
    print(out,flush=True);json.dump(out,open(f'res{seed}.json','w'))
    if seed==0:np.savez('rag.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
