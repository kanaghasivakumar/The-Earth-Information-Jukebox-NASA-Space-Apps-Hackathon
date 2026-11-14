from __future__ import annotations
import numpy as np

def normalize_percentile(values, low=5.0, high=95.0, epsilon=1e-8):
    """Map a signal to [0,1] robustly; constant/near-static signals become zero."""
    x=np.asarray(values,dtype=float)
    if x.size == 0: return x.copy()
    finite=np.isfinite(x); out=np.zeros_like(x)
    if not finite.any(): return out
    lo,hi=np.percentile(x[finite],[low,high])
    if hi-lo <= epsilon: return out
    out[finite]=np.clip((x[finite]-lo)/(hi-lo),0,1)
    return out

def _smooth(x, window):
    if int(window)!=window or window<1: raise ValueError('smoothing must be a positive integer')
    if window <= 1 or len(x)<2: return x.copy()
    window=int(window); kernel=np.ones(window)/window
    return np.convolve(np.pad(x,(window//2,(window-1)//2),mode='edge'),kernel,mode='valid')

def extract_features(timestamps, latent_means, latent_logvars=None, inputs=None, smoothing=1, flow_magnitude=None):
    """Return robust normalized temporal controls from chronological samples.

    Movement is the norm of the latent delta per second. Velocity is its signed
    projection on the selected latent direction. Flow magnitude uses input flow
    channels when supplied (N,5,H,W), otherwise movement is the proxy.
    """
    t=np.asarray(timestamps,dtype=float).reshape(-1); z=np.asarray(latent_means,dtype=float)
    if z.ndim!=2 or len(z)!=len(t): raise ValueError('timestamps and latent_means must have matching rows')
    if len(t)>1 and np.any(np.diff(t)<=0): raise ValueError('timestamps must be strictly increasing')
    if not len(t): return {'timestamps':t,'movement':t.copy(),'velocity':t.copy(),'flow_magnitude':t.copy(),'activity':t.copy()}
    if len(t)>1:
        pair_dt=np.diff(t); pair_distance=np.linalg.norm(np.diff(z,axis=0),axis=1)
        movement=np.r_[pair_distance[0],pair_distance]
        speed=np.r_[pair_distance[0]/pair_dt[0],pair_distance/pair_dt]
    else: movement=np.zeros(1); speed=np.zeros(1)
    dz=np.gradient(z,axis=0) if len(z)>1 else np.zeros_like(z)
    dt=np.gradient(t) if len(t)>1 else np.ones(1)
    if z.shape[1] > 1:
        direction=np.zeros(z.shape[1]); direction[np.argmax(np.var(z,axis=0))]=1
        velocity=(dz@direction)/np.maximum(dt,1e-8)
    else: velocity=dz[:,0]/np.maximum(dt,1e-8)
    if flow_magnitude is not None:
        flow=np.asarray(flow_magnitude,dtype=float).reshape(-1)
        if len(flow)!=len(t): raise ValueError('flow_magnitude must match timestamps')
    elif inputs is not None:
        a=np.asarray(inputs,dtype=float)
        if a.ndim!=4 or a.shape[1]<5: raise ValueError('inputs must have shape (N,5,H,W)')
        flow=np.mean(np.sqrt(np.sum(a[:,3:5]**2,axis=1)),axis=(1,2))
    else: flow=speed.copy()
    movement=_smooth(movement,smoothing); speed=_smooth(speed,smoothing); velocity=_smooth(velocity,smoothing); flow=_smooth(flow,smoothing)
    movement_norm=normalize_percentile(movement); speed_norm=normalize_percentile(speed); flow_norm=normalize_percentile(flow)
    activity=normalize_percentile(.5*movement_norm+.5*flow_norm)
    if np.nanmax(np.abs(movement),initial=0)<1e-6 and np.nanmax(np.abs(flow),initial=0)<1e-5: activity[:]=0; movement_norm[:]=0; speed_norm[:]=0; flow_norm[:]=0
    acceleration=np.r_[0.,np.abs(np.diff(speed))/np.maximum(np.diff(t),1e-8)] if len(t)>1 else np.zeros(1)
    smoothness=1-normalize_percentile(_smooth(acceleration,smoothing))
    # Each selected latent coordinate becomes an independent, centered control.
    controls=np.column_stack([normalize_percentile(z[:,i])*2-1 for i in range(z.shape[1])])
    for i in range(z.shape[1]):
        if np.ptp(z[:,i])<=1e-8: controls[:,i]=0
    if len(z)==1: controls[:]=0
    return {'timestamps':t,'movement':movement_norm,'velocity':speed_norm,'signed_velocity':normalize_percentile(velocity),'flow_magnitude':flow_norm,'activity':activity,'smoothness':smoothness,'latent_controls':controls}
