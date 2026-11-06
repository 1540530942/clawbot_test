#!/usr/bin/env python3
# 第 35 期迷你 Whisper：合成语音、对数梅尔谱、编码器—解码器转写，比较窄/宽训练、数据量、标签噪声和多任务
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

"""第 35 期玩具实验：迷你 Whisper。用正弦波合成“语音”（两种语言，每个词是一串音），算对数梅尔谱，
用编码器—解码器 Transformer 转写和翻译，解码器前面是任务词元。比较数据量、窄/宽训练分布、单任务/多任务、标签噪声。
这是玩具，不是真正的语音。"""
import sys,json,time,numpy as np
SR=4000;WIN=100;HOP=40;NM=24
EN=['zero','one','two','three','four','five','six','seven','eight','nine','red','blue','cat','dog','go','stop']
vocab=['<p>','<sot>','<eot>','<en>','<xx>','<nospeech>','<tr>','<tl>']+EN+['x'+w for w in EN]
ID={w:i for i,w in enumerate(vocab)};PAD,SOT,EOT,LEN,LXX,NOS,TR,TL=range(8);V=len(vocab);W0=8;W1=8+len(EN)
rng0=np.random.default_rng(1234)
# 每个词 3 个音节，每个音节两个共振频率（Hz）；两种语言词形不同
CODE={l:rng0.uniform(180,1500,(len(EN),3,2)) for l in ('en','xx')}
def mel_fb():
    f=np.fft.rfftfreq(WIN,1/SR);m=lambda x:2595*np.log10(1+x/700);im=lambda x:700*(10**(x/2595)-1);pts=im(np.linspace(m(80),m(SR/2),NM+2));F=np.zeros((NM,len(f)))
    for i in range(NM):
        a,b,c=pts[i:i+3];F[i]=np.maximum(0,np.minimum((f-a)/(b-a+1e-9),(c-f)/(c-b+1e-9)))
    return F
FB=mel_fb();HW=np.hanning(WIN)
def speak(r,lang,words,pitch=1.,speed=1.,noise=0.,kind='white'):
    """合成波形。pitch、speed 代表不同说话人，noise 是信噪比相关的幅度"""
    wav=[]
    for w in words:
        for s in CODE[lang][w]:
            n=int(SR*.07*speed*r.uniform(.85,1.15));t=np.arange(n)/SR;a,b=s*pitch;env=np.minimum(1,np.minimum(t,t[::-1])*60);wav.append((np.sin(2*np.pi*a*t+r.uniform(0,6))+.7*np.sin(2*np.pi*b*t+r.uniform(0,6)))*env)
        wav.append(np.zeros(int(SR*.02*speed)))
    w=np.concatenate(wav) if wav else np.zeros(SR//2)
    if noise>0:
        z=r.normal(0,1,len(w))
        if kind=='pub':z=np.convolve(z,np.ones(9)/3,'same')+.6*np.sin(2*np.pi*r.uniform(200,800)*np.arange(len(w))/SR)
        w=w+noise*z
    return w
def logmel(w):
    n=max(1,(len(w)-WIN)//HOP+1);fr=np.stack([w[i*HOP:i*HOP+WIN]*HW for i in range(n)]);P=np.abs(np.fft.rfft(fr,axis=1))**2;M=np.log10(np.maximum(P@FB.T,1e-3));return (M+1)/3-.5
DIST={'clean':dict(pitch=(1,1),speed=(1,1),noise=0.),'pitch':dict(pitch=(.8,1.25),speed=(1,1),noise=0.),'fast':dict(pitch=(1,1),speed=(.75,.75),noise=0.),'slow':dict(pitch=(1,1),speed=(1.3,1.3),noise=0.),'noisy':dict(pitch=(1,1),speed=(1,1),noise=.5),'pub':dict(pitch=(1,1),speed=(1,1),noise=.5,kind='pub'),'all':dict(pitch=(.8,1.25),speed=(.75,1.3),noise=.4,kind='pub')}
def sample(r,dist,lang='en',task='tr',lab=0.,nosp=.05,K=(2,3,4)):
    D=DIST[dist];k=int(r.choice(K));words=[int(x) for x in r.integers(0,len(EN),k)];p=r.uniform(*D['pitch']);s=r.uniform(*D['speed']);nz=D['noise']*r.random() if dist in('all',) else D['noise'];
    if r.random()<nosp:
        M=logmel(r.normal(0,.05+nz,SR//2));return M,[SOT,NOS,EOT]
    M=logmel(speak(r,lang,words,p,s,nz,D.get('kind','white')));
    if lang=='en':txt=[W0+w for w in words]
    else:txt=[(W1+w) for w in words] if task=='tr' else [W0+w for w in words]
    if lab>0:
        txt=[(int(W0+r.integers(0,len(EN))) if t<W1 else int(W1+r.integers(0,len(EN)))) if r.random()<lab else t for t in txt]
    return M,[SOT,LEN if lang=='en' else LXX,TR if task=='tr' else TL]+txt+[EOT]
def stem(M):
    """代替两层卷积：把相邻 3 帧拼起来，步长 2"""
    n=len(M);idx=np.arange(0,n,2);return np.concatenate([M[np.maximum(idx-1,0)],M[idx],M[np.minimum(idx+1,n-1)]],1)
def pad_batch(S):
    B=len(S);n=max(len(stem(m)) for m,_ in S);X=np.zeros((B,n,NM*3));Ma=np.zeros((B,n),bool);l=max(len(t) for _,t in S);Y=np.zeros((B,l),int)
    for i,(m,t) in enumerate(S):
        s=stem(m);X[i,:len(s)]=s;Ma[i,:len(s)]=True;Y[i,:len(t)]=t
    return X,Ma,Y
class W:
    def __init__(s,d,h,N,seed,maxs=200,maxt=12):
        r=np.random.default_rng(seed);s.d,s.h,s.N=d,h,N;s.ps=[];s.nm=[]
        def L(a,b,nm,sc=1.):
            t=T(r.normal(0,sc/np.sqrt(a),(a,b)));s.ps.append(t);s.nm.append(nm);return t
        s.In=L(NM*3,d,'In');s.E=L(V,d,'E');s.Pd=L(maxt,d,'Pd');p=np.arange(maxs)[:,None];t=10000.0**(-2*np.arange(d//2)/d);pe=np.zeros((maxs,d));pe[:,0::2]=np.sin(p*t);pe[:,1::2]=np.cos(p*t);s.Pe=pe
        s.enc=[{k:L(d,d,f'e{i}{k}') for k in 'qkvo'}|{'a':L(d,2*d,f'e{i}a'),'b':L(2*d,d,f'e{i}b')} for i in range(N)]
        s.dec=[{k:L(d,d,f'd{i}{k}') for k in 'qkvo'}|{'xq':L(d,d,f'd{i}xq'),'xk':L(d,d,f'd{i}xk'),'xv':L(d,d,f'd{i}xv'),'xo':L(d,d,f'd{i}xo'),'a':L(d,2*d,f'd{i}a'),'b':L(2*d,d,f'd{i}b')} for i in range(N)]
    def att(s,q,k,v,o,x,y,mask):
        B,n,d=x.v.shape;m=y.v.shape[1];hh=s.h;dk=d//hh;sp=lambda t,c:t.rs(B,c,hh,dk).t(0,2,1,3);Q=sp(x@q,n);K=sp(y@k,m);Vv=sp(y@v,m);sc=Q@K.t(0,1,3,2)*(1/np.sqrt(dk));o0=sc;sc=T(np.where(mask,o0.v,-1e9),(o0,));sc.bk=lambda g,mk=mask:(g*mk,);a=sc.softmax();s.last=a.v;return (a@Vv).t(0,2,1,3).rs(B,n,hh*dk)@o
    def encode(s,X,Ma):
        h=T(X)@s.In+T(s.Pe[:X.shape[1]]);mk=Ma[:,None,None,:]
        for l in s.enc:
            g=h.lnorm();h=h+s.att(l['q'],l['k'],l['v'],l['o'],g,g,mk);g=h.lnorm();h=h+(g@l['a']).relu()@l['b']
        return h.lnorm()
    def decode(s,H,Ma,y):
        n=y.shape[1];h=T(np.eye(V)[y])@s.E*np.sqrt(s.d)+T(np.eye(s.Pd.v.shape[0])[:n])@s.Pd;mc=np.tril(np.ones((n,n),bool))[None,None];mx=Ma[:,None,None,:]
        for l in s.dec:
            g=h.lnorm();h=h+s.att(l['q'],l['k'],l['v'],l['o'],g,g,mc);g=h.lnorm();h=h+s.att(l['xq'],l['xk'],l['xv'],l['xo'],g,H,mx);g=h.lnorm();h=h+(g@l['a']).relu()@l['b']
        return h.lnorm()@s.E.t(1,0)
    def loss(s,X,Ma,Y):
        H=s.encode(X,Ma);lg=s.decode(H,Ma,Y[:,:-1]);y=Y[:,1:];return ce(lg,y,(y!=PAD)*1.)
    def transcribe(s,M,lang,task,maxn=9):
        X,Ma,_=pad_batch([(M,[SOT])]);H=s.encode(X,Ma);y=[SOT]
        if lang is not None:y+=[LEN if lang=='en' else LXX,TR if task=='tr' else TL]
        for _ in range(maxn):
            t=int(s.decode(H,Ma,np.array([y])).v[0,-1].argmax());y.append(t)
            if t==EOT or t==NOS:break
        return y
def train(m,gen,steps,seed,lr=2e-3,B=16):
    r=np.random.default_rng(seed);o=Adam(m.ps);h=[]
    for st in range(1,steps+1):
        X,Ma,Y=pad_batch([gen(r) for _ in range(B)]);v,p,g=m.loss(X,Ma,Y);p.backward(g)
        n=np.sqrt(sum((q.g*q.g).sum() for q in m.ps if q.g is not None))
        if n>1:
            for q in m.ps:
                if q.g is not None:q.g=q.g/n
        o.step(lr*min(1,st/100)*(.3 if st>.8*steps else 1))
        if st%max(1,steps//8)==0:h.append((st,round(float(v),3)))
    return h
def wer(h,r):
    d=np.arange(len(h)+1)
    for i,x in enumerate(r,1):
        p=d.copy();d[0]=i
        for j,y in enumerate(h,1):d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))
    return d[len(h)]
def evalset(m,dist,lang,task,seed,n=60):
    r=np.random.default_rng(seed);E=0;Nt=0;ok=0
    for _ in range(n):
        M,t=sample(r,dist,lang,task,nosp=0);ref=t[3:-1];y=m.transcribe(M,lang,task);hyp=[q for q in y[3:] if q!=EOT];E+=wer(hyp,ref);Nt+=len(ref);ok+=hyp==ref
    return dict(wer=E/Nt,exact=ok/n)
def evals(m,seed,dists=('clean','pitch','fast','slow','noisy','pub','all')):
    return {d:evalset(m,d,'en','tr',seed+i)['wer'] for i,d in enumerate(dists)}
if __name__=='__main__':
    sd=int(sys.argv[1]);S=int(sys.argv[2]) if len(sys.argv)>2 else 1200;out={};t0=time.time()
    if S<=0:
        m=W(8,2,1,0);r=np.random.default_rng(0);X,Ma,Y=pad_batch([sample(r,'clean',nosp=0) for _ in range(2)]);X=X[:,:6];Ma=Ma[:,:6]
        v,p,g=m.loss(X,Ma,Y);p.backward(g)
        for q,nm in zip(m.ps,m.nm):
            if nm in('In','E','Pd','e0q','d0xk','d0a'):
                G=q.g.copy();i=np.unravel_index(int(np.argmax(abs(G))),G.shape);e=1e-6;q.v[i]+=e;a=m.loss(X,Ma,Y)[0];q.v[i]-=2*e;b=m.loss(X,Ma,Y)[0];q.v[i]+=e;print(nm,G[i],(a-b)/2/e)
        sys.exit()
    # ① 窄和宽：只听一种干净声音 vs 各种声音都听（论文表 2 的形状）
    for k,dist in (('narrow','clean'),('broad','all')):
        m=W(32,4,2,sd);h=train(m,lambda r:sample(r,dist),S,sd+1);out[k]=dict(hist=h,ev=evals(m,900+sd));print(sd,k,h[-1],out[k]['ev'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    # ② 固定数据量（表 6 形状）
    out['scale']={}
    for N in (50,400,3200):
        r0=np.random.default_rng(sd+5);D=[sample(r0,'all',lab=.05) for _ in range(N)];m=W(32,4,2,sd);train(m,lambda r:D[int(r.integers(0,N))],S,sd+2);out['scale'][str(N)]=evals(m,900+sd,('clean','pitch','noisy','all'));print(sd,'N',N,out['scale'][str(N)],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
    # ③ 标签噪声（弱监督）
    out['label']={}
    for lab in (0.,.3):
        m=W(32,4,2,sd);train(m,lambda r:sample(r,'all',lab=lab),S,sd+3);out['label'][str(lab)]=evals(m,900+sd,('clean','all'));print(sd,'label',lab,out['label'][str(lab)],round(time.time()-t0),flush=True)
    # ④ 多任务（图 9 形状）：小模型和大模型，英语和第二语言、翻译混合；英语算力相同
    out['multi']={}
    def mix(r):
        u=r.random()
        if u<.65:return sample(r,'all')
        if u<.85:return sample(r,'all','xx','tr')
        return sample(r,'all','xx','tl')
    for d,ss,tag in ((16,S,'small'),(32,S,'big')):
        a=W(d,2 if d==16 else 4,2,sd);train(a,lambda r:sample(r,'all'),int(ss*.65),sd+4);b=W(d,2 if d==16 else 4,2,sd);train(b,mix,ss,sd+4);out['multi'][tag]=dict(en_only=evalset(a,'all','en','tr',950+sd)['wer'],joint=evalset(b,'all','en','tr',950+sd)['wer'],xx=evalset(b,'all','xx','tr',960+sd)['wer'],tl=evalset(b,'all','xx','tl',970+sd)['wer']);print(sd,tag,out['multi'][tag],round(time.time()-t0),flush=True)
        if sd==0 and tag=='big':np.savez('m.npz',**{n:q.v for n,q in zip(b.nm,b.ps)},Pe=b.Pe,code_en=CODE['en'],code_xx=CODE['xx'],fb=FB)
    json.dump(out,open(f'res{sd}.json','w'),ensure_ascii=False)
