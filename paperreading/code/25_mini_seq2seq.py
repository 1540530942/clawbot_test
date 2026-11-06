#!/usr/bin/env python3
# 第 25 期迷你序列到序列翻译：手写 LSTM 反向传播，比较倒读、正读、层数、普通 RNN 和束搜索
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

"""第 25 期玩具实验：手写反向传播的 LSTM 序列到序列翻译（只依赖 NumPy）。
玩具“英语”到玩具“中文”：主动句 cat chases dog 和被动句 dog chased_by cat 都译成 猫 追 狗。
比较：正读和倒读、单层和两层、LSTM 和普通 RNN、状态大小、束搜索宽度。
初始化、梯度范数上限 5 取自论文；优化器换成 Adam。这是玩具，不代表论文结果。"""
import sys,json,time,math,collections,numpy as np
NOUN=['cat','dog','fish','bird','horse','cow','pig','sheep','wolf','fox','bear','lion']
NZH=list('猫狗鱼鸟马牛猪羊狼狐熊狮')
VERB=['chases','bites','sees','helps','pushes','pulls']
VZH=list('追咬看帮推拉')
PAD,EOS=0,1
SRC=['<p>','<e>']+NOUN+VERB+[v+'_by' for v in VERB]+['and']
TGT=['<p>','<e>']+NZH+VZH+['，']
SI={w:i for i,w in enumerate(SRC)};TI={w:i for i,w in enumerate(TGT)}
pn=1/np.arange(1,13);pn/=pn.sum()   # 名词频率服从齐夫分布，用来看生僻词
def clause(r):
    s,o=r.choice(12,2,replace=False,p=pn);v=int(r.integers(0,6));a=r.random()<.5
    src=[NOUN[s],VERB[v],NOUN[o]] if a else [NOUN[o],VERB[v]+'_by',NOUN[s]]
    return src,[NZH[s],VZH[v],NZH[o]]
def sentence(r,K):
    S=[];T=[]
    for i in range(K):
        a,b=clause(r);S+=(['and'] if i else [])+a;T+=(['，'] if i else [])+b
    return S,T
def enc_src(S,rev):
    x=[SI[w] for w in S];return (x[::-1] if rev else x)+[EOS]
def enc_tgt(T):return [TI[w] for w in T]+[EOS]
def batch(r,B,rev,Ks=(1,2,3)):
    K=int(r.choice(Ks));P=[sentence(r,K) for _ in range(B)]   # 同一批等长，和论文的分桶一样省算力
    X=[enc_src(s,rev) for s,_ in P];Y=[enc_tgt(t) for _,t in P]
    def pad(Z):
        n=max(map(len,Z));a=np.zeros((B,n),int)
        for i,z in enumerate(Z):a[i,:len(z)]=z
        return a
    return pad(X),pad(Y)
sig=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
class Cell:
    """一层 LSTM（或普通 RNN）。step 存缓存，back 反传一步。"""
    def __init__(s,di,n,r,kind):
        s.n,s.di,s.kind=n,di,kind;o=4*n if kind=='lstm' else n
        s.W=r.uniform(-.08,.08,(di+n,o));s.b=r.uniform(-.08,.08,o)
    def params(s):return [s.W,s.b]
    def step(s,x,h,c,m):
        z=np.concatenate([x,h],1);a=z@s.W+s.b
        if s.kind=='rnn':
            H=np.tanh(a);H2=m*H+(1-m)*h;return H2,c,(z,a,H,h,c,m)
        n=s.n;f,i,o,g=sig(a[:,:n]),sig(a[:,n:2*n]),sig(a[:,2*n:3*n]),np.tanh(a[:,3*n:]);C=f*c+i*g;tc=np.tanh(C);H=o*tc
        return m*H+(1-m)*h,m*C+(1-m)*c,(z,f,i,o,g,C,tc,H,h,c,m)
    def back(s,cache,dh,dc,GW,Gb):
        if s.kind=='rnn':
            z,a,H,h,c,m=cache;da=m*dh*(1-H*H);dz=da@s.W.T;GW+=z.T@da;Gb+=da.sum(0);return dz[:,:s.di],dz[:,s.di:]+(1-m)*dh,dc
        z,f,i,o,g,C,tc,H,h,c,m=cache;dH=m*dh;dC=m*dc+dH*o*(1-tc*tc)
        da=np.concatenate([dC*c*f*(1-f),dC*g*i*(1-i),dH*tc*o*(1-o),dC*i*(1-g*g)],1);dz=da@s.W.T;GW+=z.T@da;Gb+=da.sum(0)
        return dz[:,:s.di],dz[:,s.di:]+(1-m)*dh,dC*f+(1-m)*dc
class S2S:
    def __init__(s,n,N,seed,kind='lstm',emb=None):
        r=np.random.default_rng(seed);s.n,s.N,s.kind=n,N,kind;e=emb or n
        s.Es=r.uniform(-.08,.08,(len(SRC),e));s.Et=r.uniform(-.08,.08,(len(TGT),e));s.Wo=r.uniform(-.08,.08,(n,len(TGT)));s.bo=r.uniform(-.08,.08,len(TGT))
        s.enc=[Cell(e if l==0 else n,n,r,kind) for l in range(N)];s.dec=[Cell(e if l==0 else n,n,r,kind) for l in range(N)]
    def params(s):
        o=[s.Es,s.Et,s.Wo,s.bo]
        for c in s.enc+s.dec:o+=c.params()
        return o
    def run(s,cells,E,toks,mask,H,C,keep=True):
        cache=[]
        for t in range(toks.shape[1]):
            x=E[toks[:,t]];m=mask[:,t,None];cs=[]
            for l,c in enumerate(cells):
                H[l],C[l],k=c.step(x,H[l],C[l],m);cs.append(k);x=H[l]
            cache.append(cs)
        return cache
    def loss_grad(s,X,Y):
        B=len(X);H=[np.zeros((B,s.n)) for _ in range(s.N)];C=[np.zeros((B,s.n)) for _ in range(s.N)]
        ce=s.run(s.enc,s.Es,X,(X!=PAD)*1.,H,C)
        di=np.c_[np.full(B,EOS),Y[:,:-1]];dm=(Y!=PAD)*1.;cd=s.run(s.dec,s.Et,di,dm,H,C)
        P=s.params();G=[np.zeros_like(p) for p in P];gm={id(p):g for p,g in zip(P,G)};loss=0.
        dH=[np.zeros((B,s.n)) for _ in range(s.N)];dC=[np.zeros((B,s.n)) for _ in range(s.N)]
        for t in range(Y.shape[1]-1,-1,-1):
            top=cd[t][-1];h=(top[2] if s.kind=='rnn' else top[7]);lg=h@s.Wo+s.bo;lg=lg-lg.max(1,keepdims=True);q=np.exp(lg);q/=q.sum(1,keepdims=True);y=Y[:,t];m=dm[:,t]
            loss-=(m*np.log(q[np.arange(B),y]+1e-12)).sum();e=q.copy();e[np.arange(B),y]-=1;e*=m[:,None]
            gm[id(s.Wo)]+=h.T@e;gm[id(s.bo)]+=e.sum(0);dtop=e@s.Wo.T
            # 顶层输出的梯度直接加到 H（缓存里 H 未乘 mask，m=0 时 e=0，无影响）
            dH[-1]=dH[-1]+dtop
            dx=None
            for l in range(s.N-1,-1,-1):
                c=s.dec[l];dxx,dh,dc=c.back(cd[t][l],dH[l],dC[l],gm[id(c.W)],gm[id(c.b)]);dH[l],dC[l]=dh,dc
                if l>0:dH[l-1]=dH[l-1]+dxx
                else:dx=dxx
            np.add.at(gm[id(s.Et)],di[:,t],dx)
        for t in range(X.shape[1]-1,-1,-1):
            for l in range(s.N-1,-1,-1):
                c=s.enc[l];dxx,dh,dc=c.back(ce[t][l],dH[l],dC[l],gm[id(c.W)],gm[id(c.b)]);dH[l],dC[l]=dh,dc
                if l>0:dH[l-1]=dH[l-1]+dxx
                else:dx=dxx
            np.add.at(gm[id(s.Es)],X[:,t],dx)
        return loss/B,[g/B for g in G]
    def encode(s,ids):
        X=np.array([ids]);H=[np.zeros((1,s.n)) for _ in range(s.N)];C=[np.zeros((1,s.n)) for _ in range(s.N)];s.run(s.enc,s.Es,X,np.ones(X.shape),H,C);return H,C
    def dstep(s,toks,H,C):
        x=s.Et[toks];m=np.ones((len(toks),1));H=list(H);C=list(C)
        for l,c in enumerate(s.dec):
            H[l],C[l],_=c.step(x,H[l],C[l],m);x=H[l]
        lg=x@s.Wo+s.bo;lg=lg-lg.max(1,keepdims=True);q=np.exp(lg);q/=q.sum(1,keepdims=True);return np.log(q+1e-12),H,C
def beam_models(ms,ids,B,maxn=45):
    """束搜索；ms 多于一个时对概率取平均（集成）"""
    st=[m.encode(ids) for m in ms];hyp=[([],0.)];sts=[[(H,C) for H,C in st]];done=[]
    toks=np.array([EOS]);HC=[(H,C) for H,C in st];act=[([],0.)]
    for t in range(maxn):
        lq=None;new=[]
        for m,(H,C) in zip(ms,HC):
            l,H2,C2=m.dstep(toks,H,C);new.append((H2,C2));lq=np.exp(l) if lq is None else lq+np.exp(l)
        lq=np.log(lq/len(ms)+1e-12);tot=np.array([a[1] for a in act])[:,None]+lq;fl=tot.ravel();idx=np.argsort(-fl)[:B*2];keep=[];
        for k in idx:
            i,w=int(k)//len(TGT),int(k)%len(TGT)
            if w==PAD:continue
            if w==EOS:done.append((act[i][0],fl[k]))
            elif len(keep)<B:keep.append((i,w,fl[k]))
        if not keep or len(done)>=B and (done and max(d[1] for d in done)>keep[0][2]):break
        act=[(act[i][0]+[w],sc) for i,w,sc in keep];ii=[i for i,_,_ in keep];toks=np.array([w for _,w,_ in keep]);HC=[([h[ii] for h in H],[c[ii] for c in C]) for H,C in new]
    if not done:done=[(a[0],a[1]) for a in act]
    return max(done,key=lambda d:d[1])[0]
def bleu(hyps,refs):
    """语料级 BLEU-4，乘 100"""
    pn_=[];hl=sum(map(len,hyps));rl=sum(map(len,refs))
    for n in range(1,5):
        m=t=0
        for h,r in zip(hyps,refs):
            ch=collections.Counter(tuple(h[i:i+n]) for i in range(len(h)-n+1));cr=collections.Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1));m+=sum(min(v,cr[k]) for k,v in ch.items());t+=sum(ch.values())
        pn_.append(m/t if t else 0)
    if min(pn_)==0:return 0.
    bp=1 if hl>rl else math.exp(1-rl/max(hl,1));return 100*bp*math.exp(sum(math.log(p) for p in pn_)/4)
def evaluate(ms,rev,seed,Bs=(1,),Ks=(1,2,3,4,5),n=40):
    r=np.random.default_rng(seed);out={}
    data={K:[sentence(r,K) for _ in range(n)] for K in Ks}
    for B in Bs:
        hy={};
        for K in Ks:hy[K]=[beam_models(ms,enc_src(s,rev),B) for s,_ in data[K]]
        rf={K:[enc_tgt(t)[:-1] for _,t in data[K]] for K in Ks}
        out[B]=dict(byK={K:bleu(hy[K],rf[K]) for K in Ks},exact={K:float(np.mean([a==b for a,b in zip(hy[K],rf[K])])) for K in Ks},all=bleu(sum((hy[K] for K in Ks[:3]),[]),sum((rf[K] for K in Ks[:3]),[])))
    return out
def rare(ms,rev,seed,n=200):
    """每个名词是否被译对，按频率排名分组（只看 1 个分句的句子）"""
    r=np.random.default_rng(seed);hit=np.zeros(12);tot=np.zeros(12)
    for _ in range(n):
        s_,o=r.choice(12,2,replace=False);v=int(r.integers(0,6));S=[NOUN[s_],VERB[v],NOUN[o]] if r.random()<.5 else [NOUN[o],VERB[v]+'_by',NOUN[s_]];h=beam_models(ms,enc_src(S,rev),1)
        ref=[TI[NZH[s_]],TI[VZH[v]],TI[NZH[o]]]
        for k,x in ((0,s_),(2,o)):
            tot[x]+=1;hit[x]+=len(h)==3 and h[k]==ref[k]
    return (hit/np.maximum(tot,1)).tolist()
def train(m,steps,rev,seed,lr=2e-3,B=32):
    r=np.random.default_rng(seed+100);P=m.params();h=[];M1=[np.zeros_like(p) for p in P];M2=[np.zeros_like(p) for p in P]
    for st in range(1,steps+1):
        X,Y=batch(r,B,rev);L,G=m.loss_grad(X,Y);n=math.sqrt(sum((g*g).sum() for g in G))
        if n>5:G=[g*5/n for g in G]
        cur=lr*(1 if st<steps*2/3 else .5**((st-steps*2/3)/(steps/9)))
        for p,g,a,b in zip(P,G,M1,M2):     # Adam（论文用 SGD；CPU 上步数太少，SGD 学不动，见页面说明）
            a*=.9;a+=.1*g;b*=.98;b+=.02*g*g;p-=cur*(a/(1-.9**st))/(np.sqrt(b/(1-.98**st))+1e-8)
        if st%max(1,steps//20)==0:h.append((st,round(L/ (Y!=PAD).sum()*B,3)))
    return h
def repr_points(m,rev):
    """论文图 2：同一句的主动、被动，主客互换的句子。返回最后状态和名字"""
    names=[];V=[]
    for a,b in (('cat','dog'),('dog','cat')):
        for v in ('chases','bites'):
            for S in ([a,v,b],[b,v+'_by',a]):
                H,C=m.encode(enc_src(S,rev));V.append(np.concatenate(H+C,1)[0]);names.append(' '.join(S))
    return names,np.array(V)
ARMS={'rev':(48,1,'lstm',1),'fwd':(48,1,'lstm',0),'rev2':(48,2,'lstm',1),'rnn_rev':(48,1,'rnn',1),'rnn_fwd':(48,1,'rnn',0),'rev12':(12,1,'lstm',1),'fwd12':(12,1,'lstm',0)}
def save(m,f):np.savez(f,**{f'p{i}':p for i,p in enumerate(m.params())})
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 6000
    if S<=0:
        for kind,N in (('lstm',2),('rnn',1)):
            m=S2S(8,N,0,kind);r=np.random.default_rng(0);X,Y=batch(r,3,True,(1,2));L,G=m.loss_grad(X,Y);P=m.params();bad=0
            for p,g in zip(P,G):
                i=np.unravel_index(int(np.argmax(abs(g))),g.shape);e=1e-5;p[i]+=e;a=m.loss_grad(X,Y)[0];p[i]-=2*e;b=m.loss_grad(X,Y)[0];p[i]+=e;bad=max(bad,abs((a-b)/2/e-g[i])/(abs(g[i])+1e-9))
            print(kind,'最大相对误差',bad)
        sys.exit()
    out={};t0=time.time()
    for k,(n,N,kind,rev) in ARMS.items():
        m=S2S(n,N,sd,kind);h=train(m,S,rev,sd);Bs=(1,2,12) if k=='rev2' else (1,);res=evaluate([m],rev,7000+sd,Bs);out[k]=dict(hist=h,res=res,sec=round(time.time()-t0))
        if k=='rev2':
            names,V=repr_points(m,rev);out['repr']=dict(names=names,V=V.round(4).tolist());out['rare']=rare([m],rev,7100+sd);save(m,f'm_rev2_{sd}.npz')
        print(sd,k,h[-1],{B:(round(v['all'],1),[round(x,1) for x in v['byK'].values()]) for B,v in res.items()},out[k]['sec'],flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
