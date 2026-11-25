import numpy as np
from common import *

def run():
    r=rng(45);x,y,vx,vy,tx,ty=classification()
    def votes(z):
        a=z[:,0]+.7*z[:,1];b=z[:,0]+z[:,2];c=-z[:,1]+.3*z[:,3]
        return np.array([np.where(abs(t)<.4,-1,(t>0).astype(int)) for t in [a,b,c]]).T
    lv=votes(vx);lt=votes(tx)
    # Fit LF reliability on separate gold development labels, not evaluation labels.
    reliabilities=np.array([np.mean(lv[lv[:,j]>=0,j]==vy[lv[:,j]>=0]) for j in range(3)])
    weights=np.log(np.clip(reliabilities,.05,.95)/(1-np.clip(reliabilities,.05,.95)))
    def aggregate(l,w):
        active=l>=0;s=np.where(active,2*l-1,0)@w;return sigmoid(s)
    majority=aggregate(lt,np.ones(3));learned=aggregate(lt,weights)
    checks=[check(len(vy)==200 and len(ty)==200,'Gold development and evaluation are separate'),
            check(np.isfinite(weights).all(),'Finite fitted LF reliability weights'),
            check(np.all(aggregate(np.full((2,3),-1),weights)==.5),'All-abstain cases return uncertainty')]
    return {'majority_vote':measures(ty,majority),'gold_fitted_reliability':measures(ty,learned),
            'lf_reliabilities':reliabilities.tolist(),'lf_weights':weights.tolist(),
            'all_abstain_evaluation':int(np.sum(np.all(lt==-1,axis=1))),
            'checks':checks,'scope':'Supervised gold-fitted LF weighting on synthetic numeric alerts; not an unsupervised label model.'}
