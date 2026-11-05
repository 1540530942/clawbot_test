#!/usr/bin/env python3
# 第 17 期玩具实验：ReAct 循环
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

"""第 17 期玩具实验：真正的“想—搜—看”循环。
世界：12 个人，每人有一个老师和一个城市。问题：A 的老师的老师住哪。搜一个人，环境返回“他的老师、他的城市”。
四种方式：闭卷（直接答）、思维链（自己写出中间事实，不搜）、只行动（只搜）、ReAct（每步先写还剩几跳，再搜）。
闭卷和思维链在一个固定世界上背；只行动和 ReAct 每条样本都是新世界，只能靠搜。15% 的搜索返回“空”，要重搜。"""
import sys,json,time,numpy as np
PE=list('ABCDEFGHIJKL');CI=['京','沪','广','深','杭','蓉'];DG=list('012345')
vocab=['<p>','<e>','问','老师','住','哪','答',':',';','空']+DG+PE+CI
ID={w:i for i,w in enumerate(vocab)};V=len(vocab);L=96;PAD=0
def world(r):
    return {p:(PE[int(r.integers(12))],CI[int(r.integers(6))]) for p in PE}
FIX=world(np.random.default_rng(2024))
def question(a,h):return ['问',a]+['老师']*h+['住','哪',';']
def truth(W,a,h):
    x=a
    for _ in range(h):x=W[x][0]
    return x,W[x][1]
def traj(r,W,a,h,fmt,pf=.15):
    """返回 词列表, 是否模型生成（1）"""
    s=question(a,h);g=[0]*len(s)
    def add(ws,f):
        nonlocal s,g
        s=s+ws;g=g+[f]*len(ws)
    if fmt=='closed':
        add(['答',truth(W,a,h)[1],'<e>'],1);return s,g
    x=a;k=h
    while True:
        if fmt=='cot':
            add([str(k),x,W[x][0],W[x][1],';'],1)
        else:
            while True:
                add(([str(k)] if fmt=='react' else [])+[x],1)
                if r.random()<pf:add([':','空',';'],0)
                else:add([':',W[x][0],W[x][1],';'],0);break
        if k==0:break
        x=W[x][0];k-=1
    add(['答',W[x][1],'<e>'],1);return s,g
def batch(r,B,fmt,hs=(1,2,3)):
    S=[]
    for _ in range(B):
        W=FIX if fmt in('closed','cot') else world(r);S.append(traj(r,W,PE[int(r.integers(12))],int(r.choice(hs)),fmt))
    n=max(len(s) for s,_ in S);y=np.zeros((B,n),int);w=np.zeros((B,n-1))
    for i,(s,g) in enumerate(S):
        y[i,:len(s)]=[ID[t] for t in s];w[i,:len(s)-1]=g[1:]
    return y[:,:-1],y[:,1:],w
def make(seed,d=48):
    return Enc(V,L,d=d,h=4,ff=2*d,N=2,seed=seed,causal=True)
def train(fmt,seed,steps):
    m=make(seed);o=Adam(m.ps);r=np.random.default_rng(seed+9);t=time.time()
    for st in range(1,steps+1):
        x,y,w=batch(r,32,fmt);v,p,g=ce(m.lm(x),y,w);p.backward(g);m.P.g=m.P.g;o.step(min(2e-3,2e-3*st/200)*(.5*(1+np.cos(np.pi*st/steps))*.9+.1))
    return m,round(time.time()-t,1)
def nxt(m,s):
    lg=m.lm(np.array([[ID[t] for t in s]])).v[0,-1];lg[PAD]=-1e9;return vocab[int(lg.argmax())]
def run(m,fmt,W,a,h,r,pf=0.,maxact=9,force=None,log=None):
    """真实循环：模型写，遇到人名就由环境返回。返回 (答案, 原因)"""
    s=question(a,h)
    if fmt=='closed':
        t=nxt(m,s+['答']) if nxt(m,s)=='答' else None;return (t,'') if t else (None,'format')
    acts=0;seen=[]
    while len(s)<L-6:
        t=nxt(m,s)
        if t=='答':return nxt(m,s+['答']),'ok'
        if t=='<e>':return None,'stop'
        s.append(t)
        if t in PE:
            if fmt in('act','react'):
                acts+=1;seen.append(t)
                if acts>maxact:return None,'loop'
                if r.random()<pf:s+=[':','空',';']
                else:s+=[':',W[t][0],W[t][1],';']
                if log is not None:log.append(t)
            else:
                pass
    return None,'long'
def evaluate(m,fmt,n=150,seed=5,pf=0.,hs=(1,2,3,4)):
    out={}
    for wname in('fix','new'):
        for h in hs:
            if h>3 and wname=='fix':continue
            r=np.random.default_rng(seed+h);ok=0;why={}
            for _ in range(n):
                W=FIX if wname=='fix' else world(r);a=PE[int(r.integers(12))];pa,w=run(m,fmt,W,a,h,r,pf)
                if pa==truth(W,a,h)[1]:ok+=1
                elif w!='ok':why[w]=why.get(w,0)+1
            out[f'{wname}{h}']=ok/n
            if why:out[f'{wname}{h}_why']=why
    return out
def correct(m,n=150,seed=8):
    """人工改想法：把 k 改错，看 ReAct 是否听话（搜的跳数跟着变）"""
    r=np.random.default_rng(seed);follow=0
    for _ in range(n):
        W=world(r);a=PE[int(r.integers(12))];s=question(a,2)+['1'];pa=None;t0=s
        for _ in range(30):
            t=nxt(m,s)
            if t=='答':pa=nxt(m,s+['答']);break
            s.append(t)
            if t in PE:s+=[':',W[t][0],W[t][1],';']
        follow+=pa==truth(W,a,1)[1]
    return follow/n
if __name__=='__main__':
    sd=int(sys.argv[1]);steps=int(sys.argv[2]) if len(sys.argv)>2 else 3000
    if steps<=0:
        r=np.random.default_rng(0);W=world(r);print(' '.join(traj(r,W,'A',2,'react')[0]));print(' '.join(traj(r,W,'A',2,'act')[0]));print(' '.join(traj(r,W,'A',2,'cot')[0]));print(max(len(traj(r,world(r),'A',3,'react')[0]) for _ in range(200)))
        m,t=train('react',0,5);print(t,evaluate(m,'react',n=3));sys.exit()
    out={}
    for fmt in('closed','cot','act','react'):
        m,sec=train(fmt,sd,steps);res=dict(sec=sec,clean=evaluate(m,fmt));
        if fmt in('act','react'):res['noisy']=evaluate(m,fmt,pf=.3,hs=(2,3))
        if fmt=='react':res['edit']=correct(m)
        out[fmt]=res;print(sd,fmt,res,flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0:np.savez(f'm_{fmt}.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
