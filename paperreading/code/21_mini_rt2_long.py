#!/usr/bin/env python3
# 第 21 期 RT-2 加长版
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

"""第 21 期玩具实验：迷你 RT-2。一个模型同时看图、读字、写字，也写动作。
“网上数据”：两个形状的图，问“方框 在 哪”“左上 是 什么”，6 种形状都有。
“机器人数据”：图上还有一个光标，指令“去 方框”，输出动作词（dx dy，结束时 停）。只有 3 种形状。
比较：从零、只用机器人数据微调、和网上数据一起微调（共微调）。测试时闭环运行，也测另外 3 种没在机器人数据里出现的形状。"""
import sys,json,time,numpy as np
SH=['横线','竖线','方框','十字','对角线','小点'];PO=['左上','右上','左下','右下'];ROB=[0,1,2];NEW=[3,4,5]
ACT=['左','不动x','右','上','不动y','下','走','停']
VOC=['<p>','<e>','问','答','去','在','哪','是','什么']+SH+PO+ACT
ID={w:i for i,w in enumerate(VOC)};V=len(VOC);H=16;NP=16;L=NP+12;PAD=0
def shape(a,c,q,r):
    s=int(r.integers(4,7));oy=(q//2)*8;ox=(q%2)*8;x=ox+int(r.integers(0,8-s+1));y=oy+int(r.integers(0,8-s+1))
    if c==0:a[y+s//2,x:x+s]=1
    elif c==1:a[y:y+s,x+s//2]=1
    elif c==2:a[y,x:x+s]=1;a[y+s-1,x:x+s]=1;a[y:y+s,x]=1;a[y:y+s,x+s-1]=1
    elif c==3:a[y+s//2,x:x+s]=1;a[y:y+s,x+s//2]=1
    elif c==4:
        for i in range(s):a[y+i,x+i]=1
    else:a[y+s//2:y+s//2+2,x+s//2:x+s//2+2]=1
def scene(r,pool):
    c1,c2=r.choice(pool,2,replace=False);q1,q2=r.permutation(4)[:2];a=np.zeros((H,H));shape(a,int(c1),int(q1),r);shape(a,int(c2),int(q2),r);return a,int(c1),int(q1),int(c2),int(q2)
CENTER=[(1,1),(1,3),(3,1),(3,3)]  # 每个角的目标格（行、列，单位 4×4 图块的 2 倍刻度下取 0..3）
GOAL=[(0,0),(0,3),(3,0),(3,3)]
def with_cursor(a,cur):
    b=a.copy()*.6;y,x=cur;b[y*4+1:y*4+3,x*4+1:x*4+3]=1.;return b
def expert(cur,goal):
    y,x=cur;gy,gx=goal
    if (y,x)==(gy,gx):return ['停']
    return ['走',('左' if gx<x else '右' if gx>x else '不动x'),('上' if gy<y else '下' if gy>y else '不动y')]
def apply(cur,acts):
    y,x=cur
    if '左' in acts:x-=1
    if '右' in acts:x+=1
    if '上' in acts:y-=1
    if '下' in acts:y+=1
    return (int(np.clip(y,0,3)),int(np.clip(x,0,3)))
def patches(X):
    B=len(X);return X.reshape(B,4,4,4,4).transpose(0,1,3,2,4).reshape(B,NP,16)
class VLA(Enc):
    def __init__(s,seed):
        super().__init__(V,L,d=48,h=4,ff=96,N=2,seed=seed,causal=True);r=np.random.default_rng(seed+1);s.PE=T(r.normal(0,1/4,(16,48)));s.ps.append(s.PE)
    def fw(s,X,x):
        B,n=x.shape;M=np.zeros((n+NP,NP));M[:NP,:NP]=np.eye(NP);one=np.zeros((B,n+NP,V));one[:,NP:]=np.eye(V)[x]
        h=T(np.tile(M,(B,1,1)))@(T(patches(X))@s.PE)+T(one)@s.E*np.sqrt(s.d)+T(np.eye(s.L)[:n+NP])@s.P
        for l in s.layers:
            h=(h+s.attn(l,h)).lnorm();h=(h+(((h@l['a'])+l['ab']).relu()@l['b'])+l['bb']).lnorm()
        return h@s.E.t(1,0)
def pack(S):
    n=max(len(a)+len(b) for _,a,b in S);B=len(S);y=np.zeros((B,n),int);w=np.zeros((B,NP+n-1));t=np.zeros((B,NP+n-1),int);X=np.stack([s[0] for s in S])
    for i,(_,a,b) in enumerate(S):
        y[i,:len(a)+len(b)]=[ID[q] for q in a+b];w[i,NP+len(a)-1:NP+len(a)+len(b)-1]=1
    t[:,NP:]=y[:,1:];return X,y[:,:-1],t,w
def web(r,pool=range(6)):
    a,c1,q1,c2,q2=scene(r,list(pool))
    if r.random()<.5:return (a,['问',SH[c1],'在','哪','答'],[PO[q1],'<e>'])
    return (a,['问',PO[q1],'是','什么','答'],[SH[c1],'<e>'])
def robot(r,pool=ROB):
    a,c1,q1,c2,q2=scene(r,list(pool));cur=(int(r.integers(4)),int(r.integers(4)));return (with_cursor(a,cur),['去',SH[c1],'答'],expert(cur,GOAL[q1])+['<e>'])
def train(m,steps,mix,lr,r,pool=ROB):
    o=Adam(m.ps)
    for st in range(1,steps+1):
        S=[robot(r,pool) if r.random()<mix else web(r) for _ in range(32)];X,x,t,w=pack(S);v,p,g=ce(m.fw(X,x),t,w);p.backward(g);o.step(lr*min(1,st/50)*(.3 if st>.8*steps else 1))
    return m
def gen(m,X,prompt,n=4):
    ids=[ID[w] for w in prompt];o=[]
    for _ in range(n):
        lg=m.fw(X[None],np.array([ids])).v[0,-1];lg[PAD]=-1e9;k=int(lg.argmax())
        if VOC[k]=='<e>':break
        o.append(VOC[k]);ids.append(k)
    return o
def rollout(m,r,pool):
    a,c1,q1,c2,q2=scene(r,list(pool));cur=(int(r.integers(4)),int(r.integers(4)));goal=GOAL[q1]
    for _ in range(8):
        acts=gen(m,with_cursor(a,cur),['去',SH[c1],'答'])
        if '停' in acts:return cur==goal
        cur=apply(cur,acts)
    return False
def evaluate(m,seed=7,n=100):
    r=np.random.default_rng(seed);out={}
    for nm,pool in(('seen',ROB),('new',NEW)):
        out[nm]=float(np.mean([rollout(m,r,pool) for _ in range(n)]))
    return out
def vqa(m,seed=8,n=100):
    r=np.random.default_rng(seed);ok=0
    for _ in range(n):
        a,p,b=web(r);ok+=gen(m,a,p)==b[:-1]
    return ok/n
if __name__=='__main__':
    sd=int(sys.argv[1]);A=int(sys.argv[2]) if len(sys.argv)>2 else 3000;Bb=int(sys.argv[3]) if len(sys.argv)>3 else 1500
    if A<=0:
        m=VLA(0);r=np.random.default_rng(0);S=[robot(r),web(r),robot(r)];X,x,t,w=pack(S);v,p,g=ce(m.fw(X,x),t,w);p.backward(g);print(v,S[0][1:],S[1][1:]);print(rollout(m,r,ROB),evaluate(m,n=2),vqa(m,n=2));sys.exit()
    r=np.random.default_rng(sd+5);t0=time.time();m=VLA(sd);train(m,A,0,2e-3,r);W=[p.v.copy() for p in m.ps];out={'pre_vqa':vqa(m)};print(sd,'预训练 VQA',out['pre_vqa'],round(time.time()-t0),flush=True)
    def cl():
        q=VLA(sd)
        for p,v in zip(q.ps,W):p.v=v.copy()
        return q
    runs=[('从零，只用机器人',lambda:train(VLA(sd+9),Bb,1.,2e-3,r)),('预训练后只用机器人微调',lambda:train(cl(),Bb,1.,8e-4,r)),('共微调（机器人 50%）',lambda:train(cl(),2*Bb,.5,8e-4,r))]   # 第一轮 1500 步，连见过的形状都只有 4% 到 61%，明显没训练够；共微调步数翻倍，让它吃到同样多的机器人样本
    for n,f in runs:
        q=f();out[n]=dict(evaluate(q),vqa=vqa(q));print(sd,n,out[n],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
        if sd==0:np.savez('m_'+('cof' if '共' in n else 'ft' if '只用机器人微调' in n else 'sc')+'.npz',**{f'p{i}':p.v for i,p in enumerate(q.ps)})
