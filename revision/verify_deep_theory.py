"""Numerical falsification checks for the deeper mathematical derivations.
Exact identities are compared to independently evaluated quantities. Passing is
not a proof and does not establish benchmark assumptions.
"""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from scipy.linalg import solve_discrete_lyapunov
rng=np.random.default_rng(3409)
F=lambda x,e:x.T@x/(np.sum(x*x)+e)
norm=np.linalg.norm
out={}
# The second differential: automatic differentiation against the written formula.
hess_errors=[];hess_ratios=[]
for _ in range(120):
 m,d=3,5;X=rng.normal(size=(m,d));E=rng.normal(size=(m,d));eps=float(10**rng.uniform(-2,1));s=np.sum(X*X)+eps;a=np.sum(X*E)
 analytic=2*E.T@E/s-4*a*(X.T@E+E.T@X)/s**2-2*np.sum(E*E)*(X.T@X)/s**2+8*a*a*(X.T@X)/s**3
 xt=torch.tensor(X,dtype=torch.float64);et=torch.tensor(E,dtype=torch.float64)
 def fn(t):
  z=xt+t*et
  return z.T@z/(z.square().sum()+eps)
 numerical=torch.autograd.functional.hessian(lambda t:fn(t).sum(),torch.tensor(0.,dtype=torch.float64)).item()
 hess_errors.append(abs(numerical-analytic.sum()));hess_ratios.append(norm(analytic)/(20/eps*norm(E)**2))
assert max(hess_errors)<1e-11 and max(hess_ratios)<1
out['covariance_differential']={'trials':120,'max_second_differential_sum_error':max(hess_errors),'max_bound_ratio':max(hess_ratios)}
# Spherical Gaussian covariance: scalar integral vs independent Monte Carlo.
m,d=2,4;D=m*d;g=rng.normal(size=(m,d))*.35;sig=.7;eps=.23;N=norm(g)**2
Mu=lambda u:(1+2*sig**2*u)**(-D/2)*np.exp(-u*N/(1+2*sig**2*u))
alpha=quad(lambda u:np.exp(-eps*u)*Mu(u)*sig**2/(1+2*sig**2*u),0,np.inf,epsabs=1e-12)[0]
beta=quad(lambda u:np.exp(-eps*u)*Mu(u)/(1+2*sig**2*u)**2,0,np.inf,epsabs=1e-12)[0]
expected=m*alpha*np.eye(d)+beta*g.T@g
samples=g+rng.normal(size=(800000,m,d))*sig
mc=np.einsum('nmi,nmj,n->ij',samples,samples,1/(np.sum(samples*samples,axis=(1,2))+eps))/len(samples)
assert norm(mc-expected)<.003
out['Gaussian_covariance']={'samples':len(samples),'integral_MC_Frobenius_error':norm(mc-expected),'beta':beta}
# Exact stationary Lyapunov covariance for each modal regime.
modal_errors=[]
for _ in range(250):
 mu=float(rng.uniform(0,.98));delta=float(rng.uniform(.1,5));rho=float(rng.uniform(.005,.95)*2*(1+mu)/((1-mu)*delta));sigma=float(rng.uniform(.01,2))
 T=np.array([[1-rho*(1-mu)*delta,rho*mu],[-(1-mu)*delta,mu]])
 B=(1-mu)*np.array([rho,1.]);S=solve_discrete_lyapunov(T,sigma**2*np.outer(B,B));den=2*(1+mu)-rho*(1-mu)*delta
 written=sigma**2/den*np.array([[rho*(1+mu)/delta,rho*(1-mu)],[rho*(1-mu),2*(1-mu)]])
 modal_errors.append(norm(S-written)/(1+norm(S)))
assert max(modal_errors)<1e-10
out['modal_covariance']={'regimes':250,'max_relative_Lyapunov_identity_error':max(modal_errors)}
# Actual nonlinear rank-one QR map: derivative and explicit remainder bound.
remainder_ratios=[];jac_errors=[];local_contraction_ratios=[]
for _ in range(35):
 d=5;gaps=rng.uniform(.2,3,size=d-1);mu=float(rng.uniform(0,.9));rho=float(rng.uniform(.005,.7)*2*(1+mu)/((1-mu)*gaps.max()))
 def fn(z):
  x=z[:d-1];H=z[d-1:];u=np.r_[1.,x]/np.sqrt(1+np.sum(x*x));C=np.diag(np.r_[0.,-gaps]);P=np.eye(d)-np.outer(u,u)
  Hp=mu*P@H+(1-mu)*P@C@u;a=u+rho*Hp
  return np.r_[a[1:]/a[0],Hp]
 A=np.zeros((2*d-1,2*d-1))
 for j,delta in enumerate(gaps):
  hi=d+j
  A[j,j]=1-rho*(1-mu)*delta;A[j,hi]=rho*mu;A[hi,j]=-(1-mu)*delta;A[hi,hi]=mu
 h=1e-6;J=np.column_stack([(fn(np.eye(len(A))[j]*h)-fn(-np.eye(len(A))[j]*h))/(2*h) for j in range(len(A))]);jac_errors.append(norm(J-A))
 ch=mu/2+1.5*(1-mu)*gaps.max();cl=mu+(1-mu)*gaps.max();e0=min(.25,(4*rho*ch)**-.5);c2=ch+2*rho*(1.25*ch+.25*(.5+rho*ch)*cl)
 P=solve_discrete_lyapunov(A.T,np.eye(len(A)));pmax=np.linalg.eigvalsh(P).max();pmin=np.linalg.eigvalsh(P).min()
 radius=min(e0,1/(8*norm(A.T@P,2)*c2),1/(2*c2*np.sqrt(pmax)))
 for j in range(50):
  z=rng.normal(size=len(A));z*=e0*rng.uniform(.001,1)/norm(z)
  remainder_ratios.append(norm(fn(z)-A@z)/(c2*norm(z)**2))
  z=rng.normal(size=len(A));z*=np.sqrt(pmin)*radius*rng.uniform(.001,1)/np.sqrt(z@P@z)
  bound=(1-1/(2*pmax))*(z@P@z)
  local_contraction_ratios.append((fn(z)@P@fn(z))/bound)
assert max(jac_errors)<1e-8 and max(remainder_ratios)<1+1e-10 and max(local_contraction_ratios)<1+1e-10
out['rankone_actual_QR']={'parameter_regimes':35,'states_each':50,'max_Jacobian_error':max(jac_errors),'max_explicit_remainder_ratio':max(remainder_ratios),'max_Lyapunov_contraction_ratio':max(local_contraction_ratios)}
# Exact anisotropic Gaussian KL versus trace/logdet directly.
kl_errors=[];chordal_lip=[]
for _ in range(150):
 d,r,m=8,3,4;U=np.linalg.qr(rng.normal(size=(d,r)))[0];U0=np.linalg.qr(rng.normal(size=(d,r)))[0];P=U@U.T;P0=U0@U0.T;I=np.eye(d);W=rng.normal(size=(m,d))
 a,b,a0,b0=np.exp(rng.uniform(-2,1,size=4));S=a*P+b*(I-P);S0=a0*P0+b0*(I-P0);s2=norm(P-P0)**2/2
 direct=.5*(np.trace(W@np.linalg.solve(S0,W.T))+m*(np.trace(np.linalg.solve(S0,S))-d+np.linalg.slogdet(S0)[1]-np.linalg.slogdet(S)[1]))
 k=lambda z:z-1-np.log(z)
 formula=.5*(norm(W@P0)**2/a0+norm(W@(I-P0))**2/b0)+m/2*(r*k(a/a0)+(d-r)*k(b/b0)+(a-b)*(1/b0-1/a0)*s2)
 kl_errors.append(abs(direct-formula)/(1+abs(direct)))
 K=rng.normal(size=(d-r,r));Kp=K+rng.normal(size=K.shape)*.1
 def proj(K):
  X=np.vstack([np.eye(r),K]);return X@np.linalg.solve(X.T@X,X.T)
 chordal_lip.append(norm(proj(K)-proj(Kp))/np.sqrt(2)/norm(K-Kp))
assert max(kl_errors)<1e-12 and max(chordal_lip)<1+1e-12
out['orientation_and_cover']={'trials':150,'max_relative_KL_error':max(kl_errors),'max_graph_chordal_Lipschitz_ratio':max(chordal_lip)}
# Neutral-state feedback inequalities at equality versus the horizon-independent bound.
neutral_ratios=[]
for _ in range(75):
 q=float(rng.uniform(.2,.98));a=float(rng.uniform(.2,.97));B=float(rng.uniform(.01,.2));C1=float(rng.uniform(.001,.02));D1=float(rng.uniform(.001,.02));J1=float(rng.uniform(.01,.5));BW=.2;n=300;w=u=0.
 Cstar=C1/(1-a);Dstar=D1/(1-a);Jstar=J1/(1-a*a);U=np.exp(Dstar+B*Cstar/(1-q))*(Jstar+Cstar*BW/(1-q))/n;W=BW/(n*(1-q))+B*U/(1-q)
 for t in range(1500):w,u=q*w+B*u+BW/n,(1+D1*a**t)*u+C1*a**t*w+J1*a**(2*t)/n
 neutral_ratios.append(max(w/W,u/U))
assert max(neutral_ratios)<1+1e-12
out['neutral_state_feedback']={'regimes':75,'horizon':1500,'max_uniform_bound_ratio':max(neutral_ratios)}
out['scope']='Independent numerical identity/inequality checks. These do not prove the theorems or verify the benchmark assumptions.'
Path(__file__).with_name('deep_theory_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
