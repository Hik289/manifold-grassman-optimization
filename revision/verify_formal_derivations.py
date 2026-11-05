"""Independent numerical checks of algebraic identities and admissible inequalities.
These checks detect transcription/algebra errors; they do not prove the theorems.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import polar
rng=np.random.default_rng(718)

def qf(x):
 q,r=np.linalg.qr(x);s=np.sign(np.diag(r));s[s==0]=1
 return q*s

def norm(x): return np.linalg.norm(x)

def lift(U,a,X): return ((X@U)*a)@U.T

out={}
# Exact quotient chart identity, discrete contraction, and stored-frame QR bound.
chart_errors=[];contraction_ratios=[];frame_ratios=[];pullback_ratios=[]
for _ in range(150):
 d=int(rng.integers(5,15));r=int(rng.integers(1,d//2+1))
 lam=np.linspace(3.7,0,d);C=np.diag(lam);gap=lam[r-1]-lam[r]
 K=rng.normal(size=(d-r,r))*.2
 S=np.eye(r)+K.T@K;ev,B=np.linalg.eigh(S);B=(B*ev**-.5)@B.T
 U=np.vstack([np.eye(r),K])@B
 A=U.T@C@U;T=B@A@np.linalg.inv(B);LJ=norm(np.diag(lam[:r])-T,2) if False else np.linalg.norm(np.diag(lam[:r])-T,2)
 rho=.7*min(1/3.7,.5/LJ,gap/(4*3.7*LJ))
 xi=rho*(np.eye(d)-U@U.T)@C@U;Up=qf(U+xi)
 actual=Up[r:]@np.linalg.inv(Up[:r]);F=lam[r:,None]*K-K*lam[None,:r]
 predicted=K+rho*F@np.linalg.inv(np.eye(r)+rho*(np.diag(lam[:r])-T))
 chart_errors.append(norm(actual-predicted))
 contraction_ratios.append(norm(actual)/(norm(K)*(1-rho*gap/2)))
 frame_ratios.append(norm(Up-U)/(math.sqrt(3)*norm(xi)))
 # Differentiate the exact projector, not a QR implementation.
 Ut=torch.tensor(U,dtype=torch.float64);xt=torch.tensor(xi/rho,dtype=torch.float64);Ct=torch.tensor(C,dtype=torch.float64)
 st=torch.tensor(float(rng.uniform(0,2)),dtype=torch.float64,requires_grad=True)
 X=Ut+st*xt;P=X@torch.linalg.inv(torch.eye(r)+st**2*(xt.T@xt))@X.T
 phi=-torch.trace(Ct@P)/2
 first=torch.autograd.grad(phi,st,create_graph=True)[0];second=torch.autograd.grad(first,st)[0]
 pullback_ratios.append(abs(second.item())/(10*3.7*norm(xi/rho)**2))
out['geometry']={'trials':150,'graph_identity_max_error':max(chart_errors),'qr_contraction_max_ratio':max(contraction_ratios),'stored_frame_max_ratio':max(frame_ratios),'pullback_second_derivative_max_ratio':max(pullback_ratios)}
assert max(chart_errors)<1e-12 and max(contraction_ratios)<=1+1e-12 and max(frame_ratios)<=1+1e-12 and max(pullback_ratios)<=1+1e-12
# Exact actual-Adam / virtual-iterate identity, with known conditional second moment.
m,d,r=3,7,2;eta=.035;rho=.018;beta1=.8;beta2=.93;epsa=.4;epsr=.12;epsc=.2;smax=7;mu=.7
G0=10.;a=1/(G0+epsa);b=1/epsa;q=beta1/(1-beta1)
Kr=(2*math.sqrt(3)+1)*b+2*math.sqrt(3)*G0/epsa**2;Kc=2*G0/epsa**2
W=rng.normal(size=(m,d))*.2;U=qf(rng.normal(size=(d,r)));Mc=np.zeros((m,r));Vc=Mc.copy();H=np.zeros((d,r));Y=W.copy();mprev=np.zeros((m,d));Uprev=U.copy();aprev=np.zeros((m,r));qprev=0.;Vbarprev=np.zeros((m,r));Rprev=np.eye(r);cprev=0.
virtual_errors=[];metric_ratios=[];invariant_ratios=[];noise_radius=.16
for t in range(1,121):
 g=W.copy();znoise=rng.normal(size=(m,d));znoise*=noise_radius*rng.random()**(1/(m*d))/norm(znoise);G=g+znoise
 Z=G@U;M=beta1*Mc+(1-beta1)*Z;V=beta2*Vc+(1-beta2)*Z**2
 Mh=M/(1-beta1**t);Vh=V/(1-beta2**t);ct=(1-beta2)/(1-beta2**t)
 EZ2=(g@U)**2+noise_radius**2/(m*d+2)
 Vbar=(beta2*Vc+(1-beta2)*EZ2)/(1-beta2**t);abar=1/(np.sqrt(Vbar)+epsa)
 moment=Mh@U.T;actual=(Mh/(np.sqrt(Vh)+epsa))@U.T
 e=actual-lift(U,abar,moment)
 Gperp=G-(G@U)@U.T;s=min(smax,norm(G)/(norm(Gperp)+epsr));Kdir=lift(U,abar,G)+s*Gperp
 qt=beta1*(1-beta1**(t-1))/(1-beta1)
 Tprev=Mc/(1-beta1**(t-1))@U.T if t>1 else mprev
 J=qprev*lift(Uprev,aprev,mprev)-qt*lift(U,abar,Tprev)
 F=e-J
 Wnext=W-eta*(actual+s*Gperp);Ynext=Wnext-eta*qt*lift(U,abar,moment)
 virtual_errors.append(norm(Ynext-Y+eta*(Kdir+F)))
 if t>1:
  # Restrict the operator difference to all basis vectors of the old routed space.
  worst=0.
  for i in range(m):
   for j in range(r):
    X=np.zeros((m,r));X[i,j]=1;old=X@Uprev.T;new=(X@Rprev.T)@U.T
    worst=max(worst,norm(lift(U,abar,new)-lift(Uprev,aprev,old)))
  metric_ratios.append(worst/(Kr*rho+Kc*(math.sqrt(ct)+math.sqrt(cprev))))
 invariant_ratios.extend([norm(Mh)/G0,np.sum(Vh)/G0**2])
 Q=(np.eye(d)-U@U.T)@G.T@Z/(norm(G)**2+epsc)
 H=mu*(np.eye(d)-U@U.T)@H+(1-mu)*Q
 Unext=qf(U+rho*H);R=polar(Unext.T@U)[0]
 Mc=M@R.T;Vc=V@(R.T**2)
 W=Wnext;Y=Ynext;mprev=moment;Uprev=U;U=Unext;aprev=abar;qprev=qt;Vbarprev=Vbar;Rprev=R;cprev=ct
out['actual_adam']={'steps':120,'virtual_identity_max_error':max(virtual_errors),'metric_variation_max_ratio':max(metric_ratios),'state_invariant_max_ratio':max(invariant_ratios)}
assert max(virtual_errors)<1e-12 and max(metric_ratios)<1 and max(invariant_ratios)<1
# Residual coercivity: bounded spherical noise, including conditioning on parallel components.
coercive_differences=[]
for _ in range(12):
 g=rng.normal(size=(2,5))*.07;U=qf(rng.normal(size=(5,2)));P=U@U.T;v=g@(np.eye(5)-P)
 noise=rng.normal(size=(180000,2,5));noise*=((.4*rng.random(len(noise))**.1)/np.linalg.norm(noise,axis=(1,2)))[:,None,None]
 G=g+noise;E=G@(np.eye(5)-P);en=np.linalg.norm(E,axis=(1,2));gn=np.linalg.norm(G,axis=(1,2));s=np.minimum(5,gn/(en+.04))
 lhs=np.sum(v*np.mean(s[:,None,None]*E,axis=0));rhs=norm(v)**2-.04*norm(v)
 coercive_differences.append(lhs-rhs)
out['residual']={'settings':12,'samples_each':180000,'minimum_coercivity_slack_MC':min(coercive_differences)}
assert min(coercive_differences)>0
# The optimized variance must attain its analytic scalar objective minimum.
posterior_errors=[]
for _ in range(100):
 h=float(rng.uniform(.001,5));lam=float(rng.uniform(.1,20));s0=float(rng.uniform(.01,3));v=1/(s0**-2+lam*h)
 obj=.5*(h*v+(v/s0**2-1-np.log(v/s0**2))/lam)
 closed=np.log1p(lam*h*s0**2)/(2*lam)
 posterior_errors.append(abs(obj-closed))
out['posterior']={'trials':100,'optimized_objective_max_error':max(posterior_errors)}
assert max(posterior_errors)<1e-13
# Full covariance oracle versus repeated diagonal-only rotations.
cov_errors=[];cov_ratios=[];trace_errors=[]
for trial in range(20):
 r=4;B=np.zeros((r,r));v=np.zeros(r);eold=np.zeros(r);beta=.91
 for t in range(80):
  z=rng.normal(size=r);R=qf(np.eye(r)+.08*rng.normal(size=(r,r)))
  Bt=beta*B+(1-beta)*np.outer(z,z);vt=beta*v+(1-beta)*z*z
  O=Bt-np.diag(np.diag(Bt));B=R@Bt@R.T;v=(R*R)@vt
  e=np.diag(B)-v;pred=beta*(R*R)@eold+np.diag(R@O@R.T)
  cov_errors.append(norm(e-pred));trace_errors.append(abs(e.sum()))
  bound=beta*norm(eold)+2*np.linalg.norm(R-np.eye(r),2)*norm(O)
  cov_ratios.append(norm(e)/(bound+1e-30));eold=e
out['covariance_transport']={'transitions':1600,'recursion_max_error':max(cov_errors),'error_bound_max_ratio':max(cov_ratios),'trace_max_error':max(trace_errors)}
assert max(cov_errors)<1e-13 and max(cov_ratios)<1+1e-12 and max(trace_errors)<1e-12
out['interpretation']='Numerical identity and inequality checks only; no replacement for proof or empirical certification of assumptions.'
Path(__file__).with_name('formal_derivation_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
