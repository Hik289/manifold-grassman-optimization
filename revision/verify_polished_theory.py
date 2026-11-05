"""Independent numerical checks for sharper inequalities and exact witnesses.
These checks supplement proof review; they do not establish statistical assumptions.
"""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import gammaln,logsumexp,xlogy,hyp1f1
from scipy.stats import beta,binom
rng=np.random.default_rng(20261102)
N=np.linalg.norm
out={}
qr_w=torch.autograd.functional.hessian(lambda t:-.5/(1+t*t),torch.tensor(0.,dtype=torch.float64)).item()
hh_w=torch.autograd.functional.hessian(lambda t:t*t/(t*t+.4),torch.tensor(0.,dtype=torch.float64)).item()*.4/2
assert abs(qr_w-1)<1e-14 and abs(hh_w-1)<1e-14
def qf(x):
 q,r=np.linalg.qr(x);return q*np.where(np.diag(r)<0,-1,1)
# Sharp QR pullback, independently differentiated from its exact projector.
ratios=[]
for _ in range(160):
 d=int(rng.integers(4,11));r=int(rng.integers(1,d//2+1));U=qf(rng.normal(size=(d,r)));E=(np.eye(d)-U@U.T)@rng.normal(size=(d,r));C=rng.normal(size=(d,d));C=(C+C.T)/2;spread=np.ptp(np.linalg.eigvalsh(C))
 ut,et,ct=map(lambda x:torch.tensor(x,dtype=torch.float64),[U,E,C]);s=torch.tensor(rng.uniform(-2,2),dtype=torch.float64,requires_grad=True)
 X=ut+s*et;P=X@torch.linalg.inv(X.T@X)@X.T;F=-torch.trace(ct@P)/2
 a=torch.autograd.grad(F,s,create_graph=True)[0];b=torch.autograd.grad(a,s)[0]
 ratios.append(abs(b.item())/(spread*N(E)**2))
assert max(ratios)<=1+1e-10
out['QR_pullback']={'random_planes':160,'max_ratio_to_spectral_width':max(ratios),'equality_witness_ratio':qr_w}
# Householder Hessian identity and sharp compressed operator bound.
errors=[];opratios=[]
for _ in range(100):
 m,d=2,4;X=rng.normal(size=(m,d));E=rng.normal(size=(m,d));eps=10**rng.uniform(-2,1);y=np.r_[X.ravel(),np.sqrt(eps)];v=rng.normal(size=d);v/=N(v)
 B=np.zeros((m*d+1,m*d+1));B[:m*d,:m*d]=np.kron(np.eye(m),np.outer(v,v));ty=torch.tensor(y,dtype=torch.float64);tb=torch.tensor(B,dtype=torch.float64)
 H=torch.autograd.functional.hessian(lambda z:(z@tb@z)/(z@z),ty).numpy();val=y@B@y/(y@y);R=np.eye(len(y))-2*np.outer(y,y)/(y@y)
 errors.append(N(H-2/(y@y)*R@(B-val*np.eye(len(y)))@R))
 s=N(X)**2+eps;a=np.sum(X*E)
 Hmat=2*E.T@E/s-4*a*(X.T@E+E.T@X)/s**2-2*N(E)**2*(X.T@X)/s**2+8*a*a*(X.T@X)/s**3
 opratios.append(N(Hmat,2)/(2/eps*N(E)**2))
assert max(errors)<1e-11 and max(opratios)<=1+1e-10
out['Householder_compression']={'trials':100,'max_Hessian_identity_error':max(errors),'max_operator_envelope_ratio':max(opratios),'rank_one_at_zero_equality_ratio':hh_w}
# Ky Fan perturbation inequality, including its sharp second-order constant.
def excess(C,P,r):return .5*(np.linalg.eigvalsh(C)[-r:].sum()-np.trace(C@P))
fan_ratios=[];dk_ratios=[];dk_eq=[]
for _ in range(220):
 d,r=7,2;V=qf(rng.normal(size=(d,d)));vals=np.linspace(0,4,d);C=(V*vals)@V.T;U=qf(rng.normal(size=(d,r)));P=U@U.T;E=rng.normal(size=(d,d));E=(E+E.T)*.03;Cp=C+E;gap=vals[-r]-vals[-r-1];a=10**rng.uniform(-1,1)
 bound=(1+a)*excess(C,P,r)+(1+1/a)*N(E)**2/(4*gap);fan_ratios.append(excess(Cp,P,r)/bound)
 val,Vp=np.linalg.eigh(Cp);Ut=V[:,-r:];Up=Vp[:,-r:];delta=val[-r]-vals[-r-1]
 if delta>0:
  residual=N(V[:,:-r].T@E@Up);actual=N(Ut@Ut.T-Up@Up.T)/np.sqrt(2)
  dk_ratios.append(actual*delta/(residual+1e-30))
for a in [.2,1,5]:
 C=np.diag([2.,1.]);e=1e-4;E=np.array([[0,e],[e,0]]);th=-e/a;u=np.array([np.cos(th),np.sin(th)]);P=np.outer(u,u)
 dk_eq.append((excess(C+E,P,1)-(1+a)*excess(C,P,1))/N(E)**2/((1+1/a)/4))
assert max(fan_ratios)<1+1e-10 and max(dk_ratios)<1+1e-10 and max(abs(np.array(dk_eq)-1))<1e-5
out['spectral_bounds']={'trials':220,'max_Ky_Fan_bound_ratio':max(fan_ratios),'Ky_Fan_second_order_witness_ratios':dk_eq,'max_Davis_Kahan_residual_ratio':max(dk_ratios)}
# Exact Bernstein/Freedman quadratic inversions.
inversion_errors=[]
for _ in range(150):
 b=float(rng.uniform(.1,2));v=float(rng.uniform(.01,5));l=float(rng.uniform(.1,12));u=b*l/3+np.sqrt(2*v*l+b*b*l*l/9);inversion_errors.append(abs(u*u/(2*(v+b*u/3))-l))
assert max(inversion_errors)<1e-12
out['variance_sensitive_inversion']={'trials':150,'max_exponent_inversion_error':max(inversion_errors)}
# Maurer's exact moment and a genuine small-loss comparison.
def logXi(n):
 k=np.arange(n+1);z=k/n
 return logsumexp(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+xlogy(k,z)+xlogy(n-k,1-z))
me=[]
for n in [8,20,100,1000]:
 k=np.arange(n+1);z=k/n
 for p in [.13,.71]:
  logpm=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+xlogy(k,p)+xlogy(n-k,1-p)
  kl=xlogy(z,z/p)+xlogy(1-z,(1-z)/(1-p))
  me.append(abs(logsumexp(logpm+n*kl)-logXi(n)))
 assert logXi(n)<=np.log(2*np.sqrt(n))+1e-12
assert max(me)<1e-10
n=10000;K=20;delta=.05;c=(K+logXi(n)-np.log(delta))/n
out['Seeger_Maurer']={'max_Bernoulli_moment_identity_error':max(me),'zero_empirical_loss_example':{'n':n,'KL_budget':K,'delta':delta,'exact_kl_inverse':-np.expm1(-c),'Hoeffding_oracle_envelope_for_comparison':np.sqrt((K+np.log(1/delta))/(2*n))}}
# Clopper-Pearson actual binomial coverage, not just a formula mirror.
cover=[]
for m in [1,3,10,25,60]:
 k=np.arange(m+1);upper=np.r_[beta.ppf(.95,k[:-1]+1,m-k[:-1]),1.]
 for p in np.linspace(0,1,101):cover.append(np.sum(binom.pmf(k,m,p)[upper+1e-14>=p]))
assert min(cover)>=.95-1e-12
out['Clopper_Pearson']={'minimum_exact_coverage_on_grid':min(cover),'nominal_coverage':.95}
# Haar orbital entropy: exact scalar function, both asymptotics, finite-dimensional moments.
haar_errors=[];asym_errors=[];variance_errors=[]
for d in [3,8,15]:
 z=rng.normal(size=(220000,d));overlap=(z[:,0]**2)/(z*z).sum(axis=1);s2=1-overlap
 for c in [.1,1.,4.]:
  exact=hyp1f1((d-1)/2,d/2,-c);emp=np.exp(-c*s2).mean();haar_errors.append(abs(emp-exact)/exact)
 c=1e5;phi=-np.log(hyp1f1((d-1)/2,d/2,-c));asy=(d-1)/2*np.log(c)+gammaln(.5)-gammaln(d/2);asym_errors.append(abs(phi-asy))
for d,r in [(5,2),(8,3),(12,2)]:
 Z=rng.normal(size=(120000,d,r));Q=np.linalg.qr(Z)[0];overlap=(Q[:,:r,:]**2).sum(axis=(1,2));D=r*(d-r);exact=2*D*D/(d*d*(d-1)*(d+2));variance_errors.append(abs(overlap.var()-exact)/exact)
assert max(haar_errors)<.015 and max(asym_errors)<.001 and max(variance_errors)<.025
out['Haar_Gibbs_entropy']={'max_rank_one_partition_MC_relative_error':max(haar_errors),'max_large_c_asymptotic_error_at_1e5':max(asym_errors),'max_Haar_variance_relative_error':max(variance_errors)}
# Finite Haar quadrature: marginal Gaussian-mixture KL vs the optimized joint KL upper bound.
d,r,m=4,1,2;M=700;a,b=1.5,.5;W=rng.normal(size=(m,d))*.12;U=np.eye(d)[:,:r];P=U@U.T;V=rng.normal(size=(M,d));V/=N(V,axis=1)[:,None];s2=1-np.sum(V[:,:r]**2,axis=1);kap=a/b;c=m*(kap-1)**2/(2*kap)
proj=np.einsum('md,jd->jm',W,V);Kj=.5*(np.sum(proj*proj,axis=1)/a+(N(W)**2-np.sum(proj*proj,axis=1))/b)+c*s2
bound=-logsumexp(-Kj)+np.log(M)
Ssqrt=np.sqrt(b)*np.eye(d)+(np.sqrt(a)-np.sqrt(b))*P;Sinv=np.eye(d)/b+(1/a-1/b)*P
samples=W+rng.normal(size=(30000,m,d))@Ssqrt;ratios=[]
for start in range(0,len(samples),1000):
 X=samples[start:start+1000];D=X-W;logq=-.5*np.einsum('bmi,ij,bmj->b',D,Sinv,D)
 px=np.einsum('bmd,jd->bjm',X,V);logp=-.5*(np.sum(X*X,axis=(1,2))[:,None]/b+(1/a-1/b)*np.sum(px*px,axis=2));logmix=logsumexp(logp,axis=1)-np.log(M)
 ratios.extend(logq-logmix)
mc=float(np.mean(ratios));se=float(np.std(ratios)/np.sqrt(len(ratios)));assert mc<bound+4*se
out['Haar_marginal_KL']={'quadrature_orientations':M,'posterior_samples':len(samples),'marginal_KL_MC':mc,'MC_standard_error':se,'Gibbs_joint_upper_bound':bound}
# The parameter root bound has the exact scalar quadratic noise coefficient.
noise_errors=[]
for _ in range(80):
 L=float(rng.uniform(.2,3));gain=float(rng.uniform(.2,2));eta=float(rng.uniform(.01,1.9)/(L*gain));sig=float(rng.uniform(.1,2))
 ar=1-eta*gain*L
 variance=solve_discrete_lyapunov(np.array([[ar]]),np.array([[eta**2*gain**2*sig**2]]))[0,0]
 exact=L*L*variance;A=gain-L*eta*gain*gain/2
 bound=L*eta*gain*gain*sig*sig/(2*A);noise_errors.append(abs(exact-bound)/(1+abs(exact)))
assert max(noise_errors)<1e-12
root_ratios=[]
for _ in range(1200):
 aa=float(rng.uniform(.01,.5));gap=float(rng.uniform(0,3));er=float(rng.uniform(0,.2));d0=float(rng.uniform(0,1));K0=float(rng.uniform(.1,3));ee=float(rng.uniform(0,2));step=float(rng.uniform(0,1))
 C=gap+er*er/2+d0*(K0+ee)+step/2*(K0+ee)**2
 new=((ee+np.sqrt(ee*ee+4*aa*C))/(2*aa))**2
 old=4*gap/(3*aa)+2*er*er/(3*aa)+4*d0*K0/(3*aa)+2*d0*d0/(3*aa)+4*step*K0*K0/(3*aa)+4*(1/aa+.5+step)*ee*ee/(3*aa)
 root_ratios.append(new/old)
assert max(root_ratios)<=1+1e-12
out['quadratic_root_stationarity']={'quadratic_noise_regimes':80,'max_exact_noise_coefficient_error':max(noise_errors),'actual_Adam_budget_comparisons':1200,'max_root_to_previous_Young_bound_ratio':max(root_ratios)}
# Check the new unrounded, pathwise tracking recursion, including the martingale term.
track_ratios=[]
for _ in range(120):
 d,r=6,2;vals=np.linspace(.5,3,d);C=np.diag(vals);gap=vals[-r]-vals[-r-1];K=rng.normal(size=(d-r,r))*.08
 U=qf(np.vstack([K,np.eye(r)]));P=U@U.T;Q=(np.eye(d)-P)@C@U;g=-Q;R0=.6;mu=gap*np.cos(R0)**2;rho=.04;alpha=rho*mu;q=(1+alpha/2)*(1-alpha)
 z=(np.eye(d)-P)@rng.normal(size=(d,r));z*=.11/N(z);bb=(np.eye(d)-P)@rng.normal(size=(d,r));bb*=.07/N(bb)
 Up=qf(U-rho*(g+z+bb));E=rng.normal(size=(d,d));E=(E+E.T)*.001;Cp=C+E;v=N(E);LR=vals[-1]-vals[0]
 A=2*(1+alpha/2)*(LR*rho*rho*N(z)**2+rho*N(bb)**2)+(1/(2*alpha)+.25)*v*v/gap
 mart=-(1+alpha/2)*rho*np.sum(g*z)
 lhs=excess(Cp,Up@Up.T,r)-q*excess(C,P,r)-mart
 track_ratios.append(lhs/A)
assert max(track_ratios)<1+1e-10
out['unrounded_tracking']={'pathwise_trials':120,'max_remainder_to_budget_ratio':max(track_ratios)}
# Exact Gaussian multiclass conditioning against independently sampled logits.
from scipy.integrate import quad
from scipy.stats import norm
probit_errors=[];binary_errors=[]
for m in [2,4,9]:
 for h in [0.,.5,2.]:
  exact=quad(lambda t:norm.pdf(t)*(1-norm.cdf(t+h)**(m-1)),-10,10,epsabs=1e-11)[0]
  noise=rng.normal(size=(240000,m));mc=np.mean(np.max(noise[:,1:]-h,axis=1)>noise[:,0]);probit_errors.append(abs(mc-exact))
  if m==2:binary_errors.append(abs(exact-norm.sf(h/np.sqrt(2))))
 assert abs(quad(lambda t:norm.pdf(t)*(1-norm.cdf(t)**(m-1)),-10,10)[0]-(1-1/m))<1e-10
assert max(probit_errors)<.006 and max(binary_errors)<1e-11
out['Gaussian_classification']={'regimes':9,'MC_samples_each':240000,'max_exact_integral_MC_error':max(probit_errors),'max_binary_probit_identity_error':max(binary_errors)}
out['scope']='Sharp constants and algebraic identities checked numerically; composed bounds are not asserted optimal, and benchmark assumptions are not certified.'
Path(__file__).with_name('polished_theory_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
