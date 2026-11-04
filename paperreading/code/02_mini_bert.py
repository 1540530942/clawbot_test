#!/usr/bin/env python3
"""第 2 期配套：迷你 BERT。玩具语言上比较双向遮词预训练、单向预训练、不预训练。
只依赖 NumPy，运行 python3 02_mini_bert.py（2 核约 15 分钟），结果写入当前目录。
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
ANI={'小猫':('吃',['鱼','老鼠'],['屋顶','沙发']),'小狗':('啃',['骨头','肉'],['院子','门口']),'小鸟':('啄',['虫子','米粒'],['树上','窗台']),'小鱼':('吞',['水草','小虾'],['河里','池塘']),'小牛':('嚼',['青草','麦秆'],['草原','牛棚'])}
ADV=['悄悄','慢慢','开心地','偷偷','大口']
vocab=['[CLS]','[SEP]','[MASK]']+list(ANI)+ADV+['在']
for a,(v,o,p) in ANI.items(): vocab+=[v]+o+p
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);L=8;M=ID['[MASK]']
A=list(ANI)
def sent(r,bad=False):
    a=A[r.integers(5)];v,o,p=ANI[a];oo=o[r.integers(2)]
    if bad:
        b=A[(A.index(a)+1+r.integers(4))%5];oo=ANI[b][1][r.integers(2)]
    return ['[CLS]',a,ADV[r.integers(5)],'在',p[r.integers(2)],v,oo,'[SEP]']
def enc(ss): return np.array([[ID[w] for w in s] for s in ss])
def mlm_batch(r,B,rate=.15,mix=(.8,.1,.1)):
    y=enc([sent(r) for _ in range(B)]);x=y.copy();w=np.zeros(y.shape)
    for i in range(B):
        idx=[j for j in range(1,7) if r.random()<rate] or [r.integers(1,7)]
        for j in idx:
            w[i,j]=1;u=r.random()
            if u<mix[0]: x[i,j]=M
            elif u<mix[0]+mix[1]: x[i,j]=r.integers(3,V)
    return x,y,w
def pre_bert(seed,steps=1200):
    m=Enc(V,L,seed=seed);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+50);h=[]
    for st in range(1,steps+1):
        x,y,w=mlm_batch(r,64);val,p,sd=ce(m.lm(x),y,w);p.backward(sd);o.step(min(st/200,1)*3e-3*(1-st/(steps+1)))
        if st%100==0 or st==1:h.append((st,round(float(val),4)))
    return m,h
def pre_ltr(seed,steps=1200):
    m=Enc(V,L,seed=seed,causal=True);o=Adam(m.ps,b2=.999);r=np.random.default_rng(seed+50);h=[]
    for st in range(1,steps+1):
        y=enc([sent(r) for _ in range(64)]);val,p,sd=ce(m.lm(y[:,:-1]),y[:,1:]);p.backward(sd);o.step(min(st/200,1)*3e-3*(1-st/(steps+1)))
        if st%100==0 or st==1:h.append((st,round(float(val),4)))
    return m,h
def per_pos_mlm(m,n=1500):
    r=np.random.default_rng(777);y=enc([sent(r) for _ in range(n)]);o={}
    for j in range(1,7):
        x=y.copy();x[:,j]=M;lg=m.lm(x).v[:,j];o[j]=float((lg.argmax(-1)==y[:,j]).mean())
    return o
def per_pos_ltr(m,n=1500):
    r=np.random.default_rng(777);y=enc([sent(r) for _ in range(n)]);lg=m.lm(y[:,:-1]).v
    return {j:float((lg[:,j-1].argmax(-1)==y[:,j]).mean()) for j in range(1,7)}
def data(r,n):
    S=[];Y=[]
    for i in range(n):
        b=i%2==1;S.append(sent(r,bad=b));Y.append(0 if b else 1)
    return enc(S),np.array(Y)
def clone(m,cls=Enc,**kw):
    n=Enc(V,L,seed=99,causal=m.causal)
    for a,b in zip(n.ps,m.ps):a.v=b.v.copy()
    return n
def finetune(m,x,y,xt,yt,seed,steps=250,lr=3e-3):
    n=clone(m) if m is not None else Enc(V,L,seed=seed+500)
    n.rng=np.random.default_rng(seed+900);H=n.add_head(2);o=Adam(n.ps,b2=.999);pos=-1 if n.causal else 0
    for st in range(1,steps+1):
        h=n.encode(x);B=len(x);r=h.t(0,2,1)  # 取一个位置
        pick=np.zeros((B,x.shape[1],1));pick[:,pos,0]=1
        c=(r@T(pick)).rs(B,n.d)@H
        val,p,sd=ce(c,y);p.backward(sd);o.step(lr*(1-st/(steps+1)))
    h=n.encode(xt).v[:,pos];pr=(h@H.v).argmax(-1);return float((pr==yt).mean())
def forwardtest():
    # 梯度检验
    r=np.random.default_rng(0);m=Enc(V,L,d=8,h=2,ff=8,N=1,seed=0);x,y,w=mlm_batch(r,4);val,p,sd=ce(m.lm(x),y,w);p.backward(sd);worst=0
    for P in m.ps[:5]:
        i=tuple(r.integers(0,s) for s in P.v.shape);g=P.g[i];o=P.v[i];e=1e-5
        P.v[i]=o+e;a=ce(m.lm(x),y,w)[0];P.v[i]=o-e;b=ce(m.lm(x),y,w)[0];P.v[i]=o;worst=max(worst,abs((a-b)/(2*e)-g))
    return worst
if __name__=='__main__':
    print('grad',forwardtest(),'V',V,flush=True)
    r=np.random.default_rng(1);xt,yt=data(r,1000);out=dict(pos_b=[],pos_l=[],ft=[],hist={})
    for seed in range(3):
        mb,hb=pre_bert(seed);ml,hl=pre_ltr(seed);out['pos_b'].append(per_pos_mlm(mb));out['pos_l'].append(per_pos_ltr(ml))
        print(seed,out['pos_b'][-1],out['pos_l'][-1],hb[-1],hl[-1],flush=True)
        if seed==0:
            out['hist']={'b':hb,'l':hl};np.savez('bert.npz',**{f'p{i}':p.v for i,p in enumerate(mb.ps)});json.dump(vocab,open('vocab.json','w'),ensure_ascii=False)
        for nl in (20,50,200):
            rr=np.random.default_rng(seed*7+nl);x,y=data(rr,nl)
            res=dict(n=nl,seed=seed,bert=finetune(mb,x,y,xt,yt,seed),ltr=finetune(ml,x,y,xt,yt,seed),none=finetune(None,x,y,xt,yt,seed));out['ft'].append(res);print(res,flush=True)
        json.dump(out,open('res.json','w'),ensure_ascii=False)
