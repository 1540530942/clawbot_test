#!/usr/bin/env python3
# 第 48 期 Mistral 7B 玩具实验：滑动窗口的视野和共享键值头的代价
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

"""第 48 期玩具实验：滑动窗口注意力（SWA）和分组查询注意力（GQA）。一是回忆任务：序列开头有一个词，隔 D 个位置后要说出它，
验证信息每层最多走 W 步。二是字符语言模型（语料是这些论文的英文全文，可换成任意英文文本），看窗口和 kv 头数对困惑度的影响。
三是验证滚动缓存：只保留 W 个位置，和完整前向结果完全相同。这是玩具。"""
import sys,json,time,glob,re,numpy as np
def corpus():
    t=' '.join(open(f,encoding='utf-8',errors='ignore').read() for f in sorted(glob.glob('../../papers/*/compact.txt')));t=re.sub(r'\s+',' ',t.lower());t=''.join(c for c in t if 32<=ord(c)<127);t=re.sub(r'[^a-z0-9 .,;:()\-]','',t);return t
class M:
    def __init__(s,V,L,d=48,h=4,kv=4,N=2,W=None,seed=0):
        r=np.random.default_rng(seed);s.V,s.L,s.d,s.h,s.kv,s.N,s.W=V,L,d,h,kv,N,W;s.dk=d//h;s.ps=[]
        def P(a,b,sc=1.):
            t=T(r.normal(0,sc/np.sqrt(a),(a,b)));s.ps.append(t);return t
        s.E=P(V,d);s.Pos=P(L,d);s.lay=[dict(q=P(d,d),k=P(d,kv*s.dk),v=P(d,kv*s.dk),o=P(d,d),a=P(d,2*d),b=P(2*d,d,.5)) for _ in range(N)]
    def params(s):return sum(p.v.size for p in s.ps)
    def attn(s,l,x):
        B,n,_=x.v.shape;h,kv,dk=s.h,s.kv,s.dk;Q=(x@l['q']).rs(B,n,h,dk).t(0,2,1,3)
        def rep(t):
            y=t.rs(B,n,kv,dk).t(0,2,1,3);g=h//kv;o=T(np.repeat(y.v,g,axis=1),(y,))
            def bk(gr,g=g,kv=kv):
                sh=gr.shape;return (gr.reshape(sh[0],kv,g,sh[2],sh[3]).sum(2),)
            o.bk=bk;return o
        K=rep(x@l['k']);Vv=rep(x@l['v']);sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk));mask=np.tril(np.ones((n,n),bool))
        if s.W is not None:mask&=~np.tril(np.ones((n,n),bool),-s.W-1)
        o=sc;sc2=T(np.where(mask,o.v,-1e9),(o,));sc2.bk=lambda g,mask=mask:(g*mask,);a=sc2.softmax();return (a@Vv).t(0,2,1,3).rs(B,n,h*dk)@l['o']
    def fw(s,x):
        n=x.shape[1];h=T(np.eye(s.V)[x])@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n])@s.Pos
        for l in s.lay:
            h=h+s.attn(l,h.lnorm());g=h.lnorm();h=h+(g@l['a']).relu()@l['b']
        return h.lnorm()@s.E.t(1,0)
def recall(arm,D,steps,seed):
    """位置 0 是 1–8 的词，其余是 9–16 的噪声。第 D 个位置要输出位置 0 的词"""
    r=np.random.default_rng(seed);L=24;m=M(17,L,d=32,h=4,kv=4,N=arm['N'],W=arm['W'],seed=seed);o=Adam(m.ps)
    def batch(B):
        x=r.integers(9,17,(B,L));y=r.integers(1,9,B);x[:,0]=y;return x,y
    for st in range(steps):
        x,y=batch(32);lg=m.fw(x[:,:D+1]);pp=lg.softmax();sd_=np.zeros(pp.v.shape);sd_[np.arange(32),D,y]=-1/(pp.v[np.arange(32),D,y]+1e-12)/32;pp.backward(sd_);o.step(2e-3)
    x,y=batch(300);return float((m.fw(x[:,:D+1]).v[:,D].argmax(1)==y).mean())
def ppl(m,D,L,va):
    t=0;c=0
    for i in range(0,len(va),128):
        j=va[i:i+128];x=np.stack([D[a:a+L] for a in j]);lg=m.fw(x).v;e=lg[:,:-1]-lg[:,:-1].max(2,keepdims=True);lp=e-np.log(np.exp(e).sum(2,keepdims=True));t+=-np.take_along_axis(lp,x[:,1:,None],2).sum();c+=x[:,1:].size
    return float(np.exp(t/c))
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 2500;Sr=int(sys.argv[3]) if len(sys.argv)>3 else 800;t0=time.time();res={}
    res['rec']={}
    for W,N in ((4,1),(4,2),(4,4),(8,1),(8,2),(None,2)):
        row={}
        for D in (3,6,12,20):
            row[str(D)]=recall(dict(W=W,N=N),D,Sr,sd+D);print(W,N,D,row[str(D)],round(time.time()-t0),flush=True)
        res['rec'][f'{W}x{N}']=row;json.dump(res,open(f'res{sd}.json','w'))
    T_=corpus();ch=sorted(set(T_));cid={c:i for i,c in enumerate(ch)};D=np.array([cid[c] for c in T_],np.int32);L=32;r=np.random.default_rng(sd);idx=r.permutation(len(D)-L-1);tr,va=idx[:-4000],idx[-2000:]
    res['lm']={}
    for name,kw in (('基线 4 kv',dict(kv=4)),('kv=2',dict(kv=2)),('kv=1',dict(kv=1)),('窗口 16',dict(W=16)),('窗口 8',dict(W=8)),('窗口 4',dict(W=4)),('窗口 4，kv=1',dict(W=4,kv=1))):
        m=M(len(ch),L,seed=sd+1,**kw);o=Adam(m.ps);rr=np.random.default_rng(sd+2);tt=time.time()
        for st in range(1,S+1):
            j=tr[rr.integers(0,len(tr),32)];x=np.stack([D[a:a+L] for a in j]);lg=m.fw(x[:,:-1]);pp=lg.softmax();y=x[:,1:];idx2=np.indices(y.shape);sd_=np.zeros(pp.v.shape);sd_[(*idx2,y)]=-1/(pp.v[(*idx2,y)]+1e-12)/y.size;pp.backward(sd_);o.step(2e-3 if st<.8*S else 4e-4)
        k=m.kv*m.dk*2*m.N*(m.W+1 if m.W else L);res['lm'][name]=dict(ppl=ppl(m,D,L,va),params=m.params(),cache=k,sec=time.time()-tt);print(name,res['lm'][name],round(time.time()-t0),flush=True);json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False)
    res['vocab']=''.join(ch);res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False);print('done',round(time.time()-t0),flush=True)
