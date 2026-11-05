"""Numerical falsification checks and controlled geometry experiments, seed 42."""
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import expm

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(42)


def qf(a):return np.linalg.qr(a)[0]
def angles(u,v):return np.arccos(np.clip(np.linalg.svd(u.T@v,compute_uv=False),0,1))


def checks():
    ratios=[];fd=[];growth=[]
    for d,r in [(6,1),(8,3),(8,6),(32,8)]:
        for _ in range(150):
            v=qf(rng.normal(size=(d,d)));lam=np.sort(rng.uniform(.1,3,size=d))[::-1]
            c=(v*lam)@v.T;u=qf(rng.normal(size=(d,r)));us=v[:,:r]
            theta=angles(u,us);h=.5*(lam[:r].sum()-np.trace(u.T@c@u));gap=lam[r-1]-lam[r]
            grad=-(c@u-u@(u.T@c@u));bound=2*gap*math.cos(theta.max())**2*h
            ratios.append(float(np.linalg.norm(grad)**2/max(bound,1e-30)))
            s2=float(np.sin(theta)@np.sin(theta));growth.append(float(h/(.5*gap*s2)))
            xi=rng.normal(size=(d,r));xi-=u@(u.T@xi);xi/=np.linalg.norm(xi)
            eps=1e-5
            f=lambda w:-.5*np.trace(w.T@c@w)
            derivative=(f(qf(u+eps*xi))-f(qf(u-eps*xi)))/(2*eps)
            fd.append(float(abs(derivative-np.sum(grad*xi))))
    assert min(ratios)>=1-1e-8
    assert min(growth)>=1-1e-8
    assert max(fd)<1e-7
    qr=[]
    u=qf(rng.normal(size=(32,4)));xi=rng.normal(size=u.shape);xi-=u@(u.T@xi);xi/=np.linalg.norm(xi)
    a,s,b=np.linalg.svd(xi,full_matrices=False)
    for scale in np.geomspace(.001,.25,16):
        ss=scale*s
        ue=u@b.T@np.diag(np.cos(ss))@b+a@np.diag(np.sin(ss))@b
        uq=qf(u+scale*xi)
        measured=np.linalg.norm(uq@uq.T-ue@ue.T)
        upper=math.sqrt(2)/3*scale**3
        assert measured<=upper+1e-12
        qr.append({'step':float(scale),'projector_error':float(measured),'cubic_bound':upper})
    theta=np.linspace(.005,math.pi/2-.005,200)
    landscape=[{'theta':float(t),'h':.5*math.sin(t)**2,
                'gradient_squared':math.sin(t)**2*math.cos(t)**2,
                'pl_lower':2*math.cos(t)**2*.5*math.sin(t)**2,'hessian':math.cos(2*t)} for t in theta]
    return {'random_cases':600,'min_pl_ratio':min(ratios),'min_growth_ratio':min(growth),
            'max_directional_derivative_error':max(fd),'qr':qr,'landscape':landscape,
            'second_moment_rotation':{'stored_diagonal':[1.,1.],'correct_rotated_diagonal':[1.,1.],
                                      'historical_amplitude_heuristic':[2.,0.]},
            'same_batch_adam_counterexample':{'mean_gradient':.1,'expected_sign_direction':-.8,'alignment':-.08},
            'normalization_counterexample':{'raw_covariance_diagonal':[10.,.9],
                                             'normalized_covariance_diagonal':[.1,.9]}}


def tracking():
    rows=[];d=16;n=64;steps=2000
    for gap in [.05,.2,.5]:
        for drift in [0.,.001,.01]:
            for rho in [.03,.15,.5]:
                local=np.random.default_rng(42)
                u=np.zeros(d);u[0]=math.cos(.1);u[1]=math.sin(.1)
                fixed=u.copy();svd=u.copy();errors={'OnlineQR':[],'Periodic25':[],'Fixed':[]}
                vals=np.r_[1.,1.-gap,np.full(d-2,.05)]
                for t in range(steps):
                    angle=drift*t;rot=np.eye(d)
                    rot[:2,:2]=[[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]
                    target=rot[:,0]
                    g=(local.normal(size=(n,d))*np.sqrt(vals))@rot.T
                    c=g.T@g/n
                    # Exact unnormalized covariance sample: this auxiliary test
                    # intentionally excludes denominator and momentum effects.
                    q=c@u-u*(u@c@u);u=u+rho*q;u/=np.linalg.norm(u)
                    if t%25==0:svd=np.linalg.eigh(c)[1][:,-1]
                    for label,v in [('OnlineQR',u),('Periodic25',svd),('Fixed',fixed)]:
                        errors[label].append(float(1-(v@target)**2))
                rows.append({'gap':gap,'drift_radians_per_step':drift,'rho':rho,'n_rows':n,'steps':steps,
                             'mean_sin2_last500':{k:float(np.mean(v[-500:])) for k,v in errors.items()},
                             'trajectory':{k:v for k,v in errors.items()}})
    return rows


if __name__=='__main__':
    out={'seed':42,'checks':checks(),'tracking':tracking(),
         'scope':'controlled raw-covariance rank-one tracking; not end-to-end task accuracy or a population generalization certificate'}
    (HERE/'mechanism_results.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out['checks'].items() if k not in ('qr','landscape')},indent=2))
