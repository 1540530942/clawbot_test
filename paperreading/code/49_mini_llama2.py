#!/usr/bin/env python3
# 第 49 期 Llama 2 玩具实验：Ghost Attention 和拒绝采样微调
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

"""第 49 期玩具实验：Llama 2-Chat 的两个机制。一是 Ghost Attention：对话第一轮给一条指令（比如“全部反着答”），看后面几轮还记不记得。
先用“每轮都给指令”的合成对话从模型采样，再把后面几轮的指令去掉，只在最后一轮算损失。二是拒绝采样微调：每个提示采 K 条，
用奖励挑最好的当新示范，迭代几轮；看 max 和 median，看温度，也看只留上一轮和留下所有轮的差别。这是玩具。"""
import gc,sys,json,time,numpy as np
LET=list('abcdefgh');INS=['反','双','首'];GOOD=['好','佳'];vocab=['<p>','<e>','问','答','系']+INS+LET+GOOD
ID={c:i for i,c in enumerate(vocab)};V=len(vocab);PAD=0;L=84;Q,A,S=ID['问'],ID['答'],ID['系'];N3=3
def rule(u,ins):
    """用户说两个字母。默认照抄；指令：反=倒序，双=每个重复，首=只留第一个"""
    x=list(u)
    if ins==ID['反']:return x[::-1]
    if ins==ID['双']:return [x[0],x[0],x[1],x[1]]
    if ins==ID['首']:return [x[0]]
    return x
def user(r):return [int(ID[LET[i]]) for i in r.integers(0,8,2)]
def dial_sft(r,turns):
    """SFT 语料：第一轮有指令，后面照抄（这就是“忘了”）；或无指令。还有一轮带指令的句子，让模型学会规则"""
    ins=int(r.choice([ID[c] for c in INS]));s=[S,ins]
    for t in range(turns):
        u=user(r);s+=[Q]+u+[A]+rule(u,ins if t==0 else None)
    return s+[1]
def lmbatch(D):
    n=max(map(len,D));x=np.full((len(D),n),PAD)
    for i,s in enumerate(D):x[i,:len(s)]=s
    return x[:,:-1],x[:,1:],(x[:,1:]!=PAD).astype(float)
def make(seed):
    return Enc(V,L,d=48,h=4,ff=96,N=2,seed=seed,causal=True,pre=True)
def train_lm(m,gen,steps,lr,seed,bs=32):
    r=np.random.default_rng(seed);o=Adam(m.ps)
    for st in range(1,steps+1):
        x,y,w=gen(r,bs);v,p,g=ce(m.lm(x),y,w);p.backward(g);o.step(lr*(1 if st<.8*steps else .2))
        lastv=float(v);del v,p,g
        if st%10==0:gc.collect()  # 计算图里有循环引用，主动回收，否则峰值超过 700 MB
    return lastv
def greedy(m,s,n):
    x=np.array([s]);o=[]
    for _ in range(n):
        lg=m.lm(x).v[0,-1];t=int(lg[1:].argmax())+1;o.append(t);x=np.c_[x,[[t]]]
        if t==Q or t==1:break
    return o
def resp(m,s):
    o=greedy(m,s,6);a=[]
    for t in o:
        if t in(Q,1,A,S):break
        a.append(t)
    return a
def memory(m,seed,turns=(1,2,3,4,6,8),n=60):
    """第 1 轮给指令，走到第 t 轮提问，看是否还按指令答"""
    r=np.random.default_rng(seed);out={}
    for T_ in turns:
        ok=0
        for _ in range(n):
            ins=int(r.choice([ID[c] for c in INS]));s=[S,ins]
            for t in range(1,T_):
                u=user(r);s+=[Q]+u+[A]+rule(u,ins)
            u=user(r);s+=[Q]+u+[A];ok+=resp(m,s)==rule(u,ins)
        out[str(T_)]=ok/n
    return out
def gatt_data(m,r,bs):
    """GAtt：用老师（模型）在“指令放到这一轮”时采样答案，把前面几轮的指令去掉，只在最后一轮算损失"""
    D=[];W=[]
    for _ in range(bs):
        ins=int(r.choice([ID[c] for c in INS]));T_=int(r.integers(1,9));s=[S,ins]  # 第 1 轮也要留在训练里，否则会把它忘掉
        
        for t in range(T_-1):
            u=user(r);s+=[Q]+u+[A]+rule(u,ins)
        u=user(r);s+=[Q]+u+[A];b=len(s);a=resp(m,[S,ins,Q]+u+[A]);D.append(s+a+[Q]);W.append(b)
    x,y,w=lmbatch(D);w=w*0
    for i,b in enumerate(W):w[i,b-1:len(D[i])-1]=1
    return x,y,w
def reward(m_out):
    """奖励：奖励里“好/佳”的数量（有用），字母 h 严重扣分（不安全）。这里只是个函数，模型不知道"""
    g=sum(t in(ID['好'],ID['佳']) for t in m_out);return g-2.*(ID['h'] in m_out)-.2*max(0,len(m_out)-6)
def samp(m,P,r,T=1.,n=8):
    x=P.copy();B=len(P);done=np.zeros(B,bool);o=[[] for _ in range(B)]
    for _ in range(n):
        lg=m.lm(x).v[:,-1]/T;pr=np.exp(lg-lg.max(1,keepdims=True));pr[:,PAD]=0;pr/=pr.sum(1,keepdims=True);a=(pr.cumsum(1)>r.random((B,1))).argmax(1);x=np.c_[x,a]
        for i in range(B):
            if not done[i]:
                if a[i]==1:done[i]=True
                else:o[i].append(int(a[i]))
    return o
def rs_corpus(r,n):
    """初始语料：问 + 两个字母 + 答 + 随机字母和偶尔的“好”"""
    D=[]
    for _ in range(n):
        u=user(r);a=[int(ID[LET[int(r.integers(8))]]) if r.random()>.2 else ID['好'] for _ in range(int(r.integers(2,6)))];D.append([Q]+u+[A]+a+[1])
    return D
def rs(m,D0,r,iters,K,keepall,steps=300):
    """拒绝采样迭代：K 条里取奖励最高的当示范"""
    bag=[];hist=[]
    for it in range(iters):
        P=np.array([[Q]+user(r)+[A] for _ in range(200)]);Dn=[];mx=[];md=[]
        for i in range(0,200,1):
            Pk=np.tile(P[i],(K,1));out=samp(m,Pk,r);R=[reward(a) for a in out];j=int(np.argmax(R));Dn.append(P[i].tolist()+out[j]+[1]);mx.append(max(R));md.append(np.median(R))
        if keepall:bag=bag+Dn
        else:bag=Dn
        def gen(rr,bs):
            idx=rr.integers(0,len(bag),bs);x,y,w=lmbatch([bag[i] for i in idx]);w=w*(np.arange(x.shape[1])[None,:]>=3);return x,y,w
        train_lm(m,gen,steps,5e-4,int(r.integers(1e6)));P2=np.array([[Q]+user(r)+[A] for _ in range(200)]);o=samp(m,P2,r);hist.append(dict(rw=float(np.mean([reward(a) for a in o])),hb=float(np.mean([ID['h'] in a for a in o])),uniq=len({tuple(a) for a in o})/200,mx=float(np.mean(mx)),md=float(np.mean(md))))
    return hist
if __name__=='__main__':
    sd=int(sys.argv[1]);S1=int(sys.argv[2]) if len(sys.argv)>2 else 2500;t0=time.time();res={}
    def gen1(r,bs):return lmbatch([dial_sft(r,int(r.integers(1,5))) for _ in range(bs)])
    m=make(sd);train_lm(m,gen1,S1,1e-3,sd+1);res['base']=memory(m,sd+2);print('base',res['base'],round(time.time()-t0),flush=True)
    if sd==0:np.savez('m_base.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
    teacher=make(sd)
    for tp,mp in zip(teacher.ps,m.ps):tp.v=mp.v.copy()
    # 老师必须是冻结的副本：第一轮实验用正在训练的模型当老师，它边学边把教材带坏，全部归零（证据里留着）
    def gen2(r,bs):return gatt_data(teacher,r,bs)
    w0=[p.v.copy() for p in m.ps];train_lm(m,gen2,S1//2,5e-4,sd+3);res['gatt']=memory(m,sd+2);print('gatt',res['gatt'],round(time.time()-t0),flush=True)
    if sd==0:np.savez('m_gatt.npz',**{f'p{i}':p.v for i,p in enumerate(m.ps)})
    # 拒绝采样
    r=np.random.default_rng(sd+5);D0=rs_corpus(r,3000);ml=make(sd+6)
    def gen3(rr,bs):
        idx=rr.integers(0,len(D0),bs);return lmbatch([D0[i] for i in idx])
    train_lm(ml,gen3,1500,1e-3,sd+7);w=[p.v.copy() for p in ml.ps];res['Ncurve']={}
    P=np.array([[Q]+user(r)+[A] for _ in range(60)]);Ns=(1,2,4,8,16,32,64,100)
    for T_ in (.5,1.,1.5):
        o=[];
        for p in P:
            a=samp(ml,np.tile(p,(100,1)),r,T=T_);o.append([reward(x) for x in a])
        o=np.array(o);res['Ncurve'][str(T_)]=dict(mx={str(n):float(o[:,:n].max(1).mean()) for n in Ns},md=float(np.median(o,axis=1).mean()))
    print(res['Ncurve'],round(time.time()-t0),flush=True);res['rs']={}
    for nm,ka in (('last',False),('all',True)):
        for p,v in zip(ml.ps,w):p.v=v.copy()
        res['rs'][nm]=rs(ml,D0,np.random.default_rng(sd+8),5,16,ka);print(nm,res['rs'][nm],round(time.time()-t0),flush=True)
    res['sec']=time.time()-t0;json.dump(res,open(f'res{sd}.json','w'),ensure_ascii=False);print('done',round(time.time()-t0),flush=True)
