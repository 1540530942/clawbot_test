#!/usr/bin/env python3
# 第 36 期迷你 SAM（14000 步，数据引擎段 3500 步）
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

"""第 36 期玩具实验：迷你 SAM。图上有嵌套的形状（整体、部分、小部分），点一下可能指其中任何一层。
图像编码器只跑一次；提示编码器和掩码解码器很轻。比较：3 个输出取最小损失、只有 1 个输出；预测 IoU 排序；
全自动网格 + 筛选 + NMS 生成伪标签，再用伪标签训练。这是玩具，不代表论文结果。"""
import sys,json,time,numpy as np
S=24;PATCH=4;G_=S//PATCH;NP=G_*G_
def shape_mask(kind,x0,y0,x1,y1):
    yy,xx=np.mgrid[:S,:S];m=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    if kind=='circle':
        cx,cy,rx,ry=(x0+x1-1)/2,(y0+y1-1)/2,(x1-x0)/2,(y1-y0)/2;m=((xx-cx)/max(rx,.5))**2+((yy-cy)/max(ry,.5))**2<=1
    if kind=='tri':
        m=m&(((yy-y0)/max(y1-y0,1))>=np.abs(xx-(x0+x1-1)/2)/(max(x1-x0,1)/2+1e-6))
    return m
def scene(r,kinds=('rect','circle')):
    """返回图、嵌套组：每组从外到内 2–3 个掩码"""
    img=r.normal(0,.05,(S,S));groups=[];occ=np.zeros((S,S),bool)
    for _ in range(int(r.integers(1,3))):
        for tr in range(20):
            w,h=r.integers(9,14,2);x0,y0=r.integers(0,S-w+1),r.integers(0,S-h+1);m=shape_mask(r.choice(kinds),x0,y0,x0+w,y0+h)
            if not (m&occ).any():break
        else:continue
        occ|=m;c=r.uniform(.4,1.)*r.choice([-1,1]);img=np.where(m,c+r.normal(0,.05,(S,S)),img);ms=[m]
        ix0,iy0,ix1,iy1=x0+2,y0+2,x0+w-2,y0+h-2;pw,ph=max(4,(ix1-ix0)*2//3),max(4,(iy1-iy0)*2//3);px,py=r.integers(ix0,max(ix0+1,ix1-pw+1)),r.integers(iy0,max(iy0+1,iy1-ph+1));pm=shape_mask(r.choice(kinds),px,py,px+pw,py+ph)&m
        c2=c-np.sign(c)*r.uniform(.5,.9);img=np.where(pm,c2+r.normal(0,.05,(S,S)),img);ms.append(pm)
        if r.random()<.7:
            sw,sh=max(2,pw//2),max(2,ph//2);sx,sy=r.integers(px,max(px+1,px+pw-sw+1)),r.integers(py,max(py+1,py+ph-sh+1));sm=shape_mask('rect',sx,sy,sx+sw,sy+sh)&pm
            if sm.sum()>=2:
                c3=c2+np.sign(c)*r.uniform(.6,1.);img=np.where(sm,c3+r.normal(0,.05,(S,S)),img);ms.append(sm)
        groups.append(ms)
    return img,groups
def pe(x,y,d):
    """点的位置编码（正弦）"""
    f=2.**np.arange(d//4);a=np.array([x,y])/S*2*np.pi;return np.concatenate([np.sin(a[0]*f),np.cos(a[0]*f),np.sin(a[1]*f),np.cos(a[1]*f)])[:d]
class SAM:
    def __init__(s,seed,d=48,K=3,N=2):
        r=np.random.default_rng(seed);s.d,s.K,s.N=d,K,N;s.ps=[];s.nm=[];c=16;s.c=c
        def L(a,b,nm,sc=1.):
            t=T(r.normal(0,sc/np.sqrt(a),(a,b)));s.ps.append(t);s.nm.append(nm);return t
        s.In=L(PATCH*PATCH,d,'In');s.Pp=L(NP,d,'Pp',.1)
        s.enc=[{k:L(d,d,f'e{i}{k}') for k in 'qkvo'}|{'a':L(d,2*d,f'e{i}a'),'b':L(2*d,d,f'e{i}b')} for i in range(N)]
        s.fg=L(1,d,'fg',.1);s.out=L(K+1,d,'out',.1)   # K 个掩码词元 + 1 个 IoU 词元
        s.dec=[{k:L(d,d,f'd{i}{k}') for k in ('sq','sk','sv','so','xq','xk','xv','xo','iq','ik','iv','io')}|{'a':L(d,2*d,f'd{i}a'),'b':L(2*d,d,f'd{i}b')} for i in range(2)]
        s.up=L(d,c*PATCH*PATCH,'up',.15);s.mh=L(d,c,'mh',.15);s.iou=L(d,K,'iou')
    def attn(s,q,k,v,o,x,y,h=4):
        B,n,d=x.v.shape;m=y.v.shape[1];dk=d//h;sp=lambda t,c:t.rs(B,c,h,dk).t(0,2,1,3);Q=sp(x@q,n);K=sp(y@k,m);V=sp(y@v,m);a=(Q@K.t(0,1,3,2)*(1/np.sqrt(dk))).softmax();return (a@V).t(0,2,1,3).rs(B,n,h*dk)@o
    def encode(s,img):
        """(B,S,S) → (B,NP,d)，每张图只跑一次"""
        B=len(img);x=img.reshape(B,G_,PATCH,G_,PATCH).transpose(0,1,3,2,4).reshape(B,NP,-1);h=T(x)@s.In+s.Pp
        for l in s.enc:
            g=h.lnorm();h=h+s.attn(l['q'],l['k'],l['v'],l['o'],g,g);g=h.lnorm();h=h+(g@l['a']).relu()@l['b']
        return h.lnorm()
    def decode(s,E,pts):
        """E:(B,NP,d) 图像嵌入；pts:(B,2)。返回 logits (B,K,S*S), iou (B,K)"""
        B=E.v.shape[0];pv=np.stack([pe(x,y,s.d) for x,y in pts])[:,None,:];tok=T(pv)+s.fg
        out=T(np.ones((B,1,1)))*s.out;t=T(np.concatenate([tok.v,out.v],1),(tok,out));nt=1;t.bk=lambda g,n=1:(g[:,:n],g[:,n:]);I=E
        for l in s.dec:
            g=t.lnorm();t=t+s.attn(l['sq'],l['sk'],l['sv'],l['so'],g,g);g=t.lnorm();t=t+s.attn(l['xq'],l['xk'],l['xv'],l['xo'],g,I);g=t.lnorm();t=t+(g@l['a']).relu()@l['b'];I=I+s.attn(l['iq'],l['ik'],l['iv'],l['io'],I.lnorm(),t.lnorm())
        U=(I@s.up).rs(B,G_,G_,PATCH,PATCH,s.c).t(0,1,3,2,4,5).rs(B,S*S,s.c);tk=t@s.mh
        mt=tk.rs(B,1+s.K+1,s.c);sel=T(mt.v[:,1:1+s.K],(mt,))
        def bk(g,sh=mt.v.shape,K=s.K):
            o=np.zeros(sh);o[:,1:1+K]=g;return (o,)
        sel.bk=bk
        logit=sel@U.t(0,2,1)
        it=T(t.v[:,1+s.K],(t,))
        def bk2(g,sh=t.v.shape,K=s.K):
            o=np.zeros(sh);o[:,1+K]=g;return (o,)
        it.bk=bk2;return logit,(it@s.iou)
def focal_dice(lg,y):
    """逐掩码的 focal + dice（论文 20:1）。lg:(K,n)，y:(n,)。返回 损失 (K,) 和对 lg 的梯度 (K,n)"""
    p=1/(1+np.exp(-np.clip(lg,-20,20)));pt=p*y+(1-p)*(1-y);a=.25*y+.75*(1-y);f=-(a*(1-pt)**2*np.log(pt+1e-9));df=(a*(1-pt)**2/(pt+1e-9)-a*2*(1-pt)*np.log(pt+1e-9)*-1*-1)*(-(2*y-1))*p*(1-p);df=(-(a*(1-pt)**2/(pt+1e-9))+2*a*(1-pt)*np.log(pt+1e-9))*(2*y-1)*p*(1-p)
    num=2*(p*y).sum(1)+1;den=p.sum(1)+y.sum()+1;d=1-num/den;dd=(-(2*y)/den[:,None]+(num/den**2)[:,None])*p*(1-p);return 20*f.mean(1)+d,20*df/y.size+dd
def iou(a,b):
    i=(a&b).sum();u=(a|b).sum();return i/u if u else 0.
def batch(r,B,kinds,pool=None):
    imgs=[];pts=[];ys=[];stage=[]
    for _ in range(B):
        if pool is not None:
            img,masks=pool[int(r.integers(0,len(pool)))];m=masks[int(r.integers(0,len(masks)))].reshape(S,S)   # auto_masks 给的是展平的
        else:
            img,gr=scene(r,kinds);g=gr[int(r.integers(0,len(gr)))];m=g[int(r.integers(0,len(g)))]
        yy,xx=np.where(m);k=int(r.integers(0,len(xx)));imgs.append(img);pts.append((xx[k],yy[k]));ys.append(m.reshape(-1)*1.)
    return np.stack(imgs),pts,ys
def step(m,o,r,B,kinds,lr,pool=None):
    imgs,pts,ys=batch(r,B,kinds,pool);E=m.encode(imgs);lg,ii=m.decode(E,pts);G=np.zeros(lg.v.shape);Gi=np.zeros(ii.v.shape);tot=0.
    for b in range(B):
        L,dL=focal_dice(lg.v[b],ys[b]);k=int(L.argmin());tot+=L[k];G[b,k]=dL[k]/B
        for j in range(m.K):
            pr=lg.v[b,j]>0;t=iou(pr,ys[b]>0);Gi[b,j]=2*(ii.v[b,j]-t)/B/m.K
    root=T(0.,(lg,ii));root.bk=lambda g:(G,Gi);root.backward();o.step(lr);return tot/B
def train(m,r,steps,kinds=('rect','circle'),pool=None,B=10,lr=7e-4):
    o=Adam(m.ps);h=[]
    for st in range(1,steps+1):
        v=step(m,o,r,B,kinds,lr*min(1,st/100)*(.1 if st>.9*steps else .3 if st>.7*steps else 1),pool)
        if st%max(1,steps//8)==0:h.append((st,round(float(v),3)))
    return h
def evaluate(m,seed,kinds,n=80):
    """一个前景点。返回 排序第一的 mIoU、三个中最好的（oracle）、IoU 头相关度"""
    r=np.random.default_rng(seed);top=[];orc=[];pi=[];ti=[]
    imgs=[];pts=[];ys=[]
    for _ in range(n):
        img,gr=scene(r,kinds);g=gr[int(r.integers(0,len(gr)))];mk=g[int(r.integers(0,len(g)))];yy,xx=np.where(mk);k=int(r.integers(0,len(xx)));imgs.append(img);pts.append((xx[k],yy[k]));ys.append(mk.reshape(-1))
    E=m.encode(np.stack(imgs));lg,ii=m.decode(E,pts)
    for b in range(n):
        us=[iou(lg.v[b,j]>0,ys[b]) for j in range(m.K)];k=int(ii.v[b].argmax());top.append(us[k]);orc.append(max(us));pi+=list(ii.v[b]);ti+=us
    return dict(top=float(np.mean(top)),oracle=float(np.mean(orc)),iouc=float(np.corrcoef(pi,ti)[0,1]) if m.K>1 else 0.)
def auto_masks(m,img,grid=6,thr=.4,stab=.8):
    """全自动：网格点 → 掩码 → 预测 IoU 和稳定性筛选 → NMS"""
    gs=np.linspace(2,S-3,grid).astype(int);pts=[(x,y) for y in gs for x in gs];E=m.encode(img[None]);Ee=T(np.repeat(E.v,len(pts),0));lg,ii=m.decode(Ee,pts);c=[]
    for b in range(len(pts)):
        for j in range(m.K):
            if ii.v[b,j]<thr:continue
            a=lg.v[b,j]>1;d=lg.v[b,j]>-1;st=iou(a,d)
            if st<stab or d.sum()<4 or d.sum()>S*S*.9 or (lg.v[b,j]>0).sum()<2:continue
            c.append((float(ii.v[b,j]),lg.v[b,j]>0))
    c.sort(key=lambda t:-t[0]);keep=[]
    for s_,mk in c:
        if all(iou(mk,k)<.7 for k in keep):keep.append(mk)
    return keep
def auto_eval(m,seed,kinds,n=30):
    r=np.random.default_rng(seed);rec=[];cnt=[];prec=[]
    for _ in range(n):
        img,gr=scene(r,kinds);ks=auto_masks(m,img);gt=[x.reshape(-1) for g in gr for x in g];cnt.append(len(ks))
        rec+=[max([iou(k,g) for k in ks],default=0)>=.7 for g in gt];prec+=[max(iou(k,g) for g in gt)>=.7 for k in ks]
    return dict(recall=float(np.mean(rec)),precision=float(np.mean(prec)) if prec else 0.,count=float(np.mean(cnt)))
if __name__=='__main__':
    sd=int(sys.argv[1]);Sx=int(sys.argv[2]) if len(sys.argv)>2 else 2500;out={};t0=time.time();r=np.random.default_rng(sd+1)
    if Sx<=0:
        m=SAM(0,d=16,K=2,N=1);imgs,pts,ys=batch(r,2,('rect',))
        def fn():
            E=m.encode(imgs);lg,ii=m.decode(E,pts);return lg,ii
        lg,ii=fn();t=[focal_dice(lg.v[b],ys[b]) for b in range(2)];G=np.stack([x[1] for x in t])
        root=T(0.,(lg,ii));root.bk=lambda g:(G,ii.v*0+1);root.backward()
        def L():
            lg,ii=fn();return sum(focal_dice(lg.v[b],ys[b])[0].sum() for b in range(2))+ii.v.sum()
        for nm,q in zip(m.nm,m.ps):
            if nm in('In','e0q','out','d0xq','d1iq','up','mh','iou','fg'):
                Gq=q.g.copy();i=np.unravel_index(int(np.argmax(abs(Gq))),Gq.shape);e=1e-6;q.v[i]+=e;a=L();q.v[i]-=2*e;b=L();q.v[i]+=e;print(nm,Gq[i],(a-b)/2/e)
        sys.exit()
    # 3 输出和 1 输出的结果来自 36c（14000 步）；36c 在数据引擎段因掩码形状的 bug 退出，这里只补跑这一段
    out=json.load(open(f'../36c/res{sd}.json'))
    # 数据引擎：先用少量人工标注（500 张图）训练，再全自动标注 2000 张，用伪标签从头训练
    D=[];rr=np.random.default_rng(sd+70)
    for _ in range(500):
        img,gr=scene(rr);D.append((img,[x for g in gr for x in g]))
    m2=SAM(sd,K=3);train(m2,r,Sx//4,pool=D);out['manual']=dict(seen=evaluate(m2,900+sd,('rect','circle')));P=[]
    for _ in range(2000):
        img,gr=scene(rr);ks=auto_masks(m2,img)
        if ks:P.append((img,ks))
    out['pool']=dict(images=len(P),masks=int(sum(len(k) for _,k in P)));print(sd,'伪标签',out['pool'],round(time.time()-t0),flush=True)
    
    if P:m3=SAM(sd,K=3);train(m3,r,Sx//4,pool=P);out['auto_only']=dict(seen=evaluate(m3,900+sd,('rect','circle')))
    else:out['auto_only']=None   # 模型没有产出任何过筛的伪标签
    m4=SAM(sd,K=3);train(m4,r,Sx//4,pool=D+P);out['both']=dict(seen=evaluate(m4,900+sd,('rect','circle')));print(sd,'manual',out['manual'],'auto',out['auto_only'],'both',out['both'],round(time.time()-t0),flush=True);json.dump(out,open(f'res{sd}.json','w'))
