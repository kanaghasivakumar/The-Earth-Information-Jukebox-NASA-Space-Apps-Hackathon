"""Deterministic latent clustering and scale-safe motif memory.

Typical use::

    memory = LatentMemory(seed=17).fit(mu, seconds, n_clusters=5)
    cluster = int(memory.labels[i])
    motif = memory.replay(cluster, variation_strength=0.15)

Inputs are chronological VAE means of any dimension. Coordinates are standardized
before K-means so large-range dimensions do not dominate. Cluster changes are
smoothed using timestamp-based minimum dwell and candidate hysteresis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class MotifNote:
    """One scale-relative event; ``beat_offset`` is relative to motif start."""
    scale_degree: int
    beat_offset: float
    duration_beats: float
    velocity: int

    def __post_init__(self):
        if self.scale_degree<0: raise ValueError('scale_degree must be nonnegative')
        if not np.isfinite(self.beat_offset) or self.beat_offset<0: raise ValueError('beat_offset must be finite and nonnegative')
        if not np.isfinite(self.duration_beats) or self.duration_beats<=0: raise ValueError('duration_beats must be positive and finite')
        if not 1<=self.velocity<=127: raise ValueError('velocity must be in [1,127]')

@dataclass(frozen=True)
class Motif:
    notes: tuple[MotifNote,...]
    name: str='motif'

    def __post_init__(self):
        if not self.notes: raise ValueError('motifs need at least one note')
        if tuple(sorted(self.notes,key=lambda n:n.beat_offset))!=self.notes: raise ValueError('motif notes must be chronological')

@dataclass(frozen=True)
class ClusterMemory:
    cluster_id: int
    center: tuple[float,...]
    sample_count: int
    motif: Motif

@dataclass(frozen=True)
class ClusterRun:
    cluster_id: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float

@dataclass(frozen=True)
class ClusterTransition:
    index: int
    time: float
    from_cluster: int
    to_cluster: int
    is_return: bool

class LatentMemory:
    """K-means memories for latent patterns, with deterministic motif replay."""
    def __init__(self, seed: int=0):
        self.seed=int(seed); self.labels_raw=np.array([],dtype=np.int64); self.labels=np.array([],dtype=np.int64)
        self.timestamps=np.array([],dtype=float); self.centers=np.empty((0,0),dtype=float)
        self.mean_=np.array([],dtype=float); self.scale_=np.array([],dtype=float)
        self.clusters: tuple[ClusterMemory,...]=(); self.n_clusters=0; self.requested_clusters=0
        self.runs: tuple[ClusterRun,...]=(); self.transitions: tuple[ClusterTransition,...]=()
        self.transition_times=np.array([],dtype=float); self.occurrences: Mapping[int,tuple[ClusterRun,...]]={}
        self.active_motifs: tuple[Motif,...]=(); self.motif_returns: tuple[ClusterTransition,...]=()

    def fit(self, latent_means: np.ndarray, timestamps: Sequence[float], *, n_clusters: int=4,
            min_dwell: float=1.0, hysteresis: float=.5,
            motif_library: Mapping[int,Motif|Sequence[MotifNote]]|None=None) -> "LatentMemory":
        """Cluster one chronological trajectory and associate a motif per cluster.

        ``n_clusters`` is reduced to the number of distinct normalized samples;
        constant and very short trajectories therefore remain valid.
        """
        z=np.asarray(latent_means,dtype=float); t=np.asarray(timestamps,dtype=float).reshape(-1)
        if z.ndim!=2 or z.shape[0]!=len(t) or z.shape[1]==0: raise ValueError('latent_means must be N by D and match timestamps')
        if not len(t): raise ValueError('latent trajectory must not be empty')
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(t)): raise ValueError('latents and timestamps must be finite')
        if len(t)>1 and np.any(np.diff(t)<=0): raise ValueError('timestamps must be strictly increasing')
        if int(n_clusters)!=n_clusters or n_clusters<1: raise ValueError('n_clusters must be a positive integer')
        if not np.isfinite(min_dwell) or min_dwell<0 or not np.isfinite(hysteresis) or hysteresis<0: raise ValueError('dwell and hysteresis must be finite and nonnegative')
        self.timestamps=t.copy(); self.requested_clusters=int(n_clusters)
        self.mean_=z.mean(axis=0); raw_scale=z.std(axis=0); self.scale_=np.where(raw_scale<1e-12,1.,raw_scale)
        normalized=(z-self.mean_)/self.scale_
        unique=np.unique(normalized,axis=0); k=min(int(n_clusters),len(unique),len(z)); self.n_clusters=k
        centers,labels=_kmeans(normalized,k,self.seed)
        self.centers=centers*self.scale_+self.mean_; self.labels_raw=labels
        self.labels=_dwell_filter(labels,t,min_dwell,hysteresis)
        memories=[]
        for cluster_id in range(k):
            count=int(np.sum(self.labels==cluster_id))
            provided=motif_library.get(cluster_id) if motif_library is not None else None
            if provided is None: motif=_default_motif(cluster_id,centers[cluster_id],seed=self.seed)
            elif isinstance(provided,Motif): motif=provided
            else: motif=Motif(tuple(provided),name=f'cluster-{cluster_id}')
            memories.append(ClusterMemory(cluster_id,tuple(map(float,self.centers[cluster_id])),count,motif))
        self.clusters=tuple(memories)
        self.runs=_cluster_runs(self.labels,t)
        seen={int(self.labels[0])}; transitions=[]
        for i in range(1,len(self.labels)):
            before,after=int(self.labels[i-1]),int(self.labels[i])
            if before!=after:
                transition=ClusterTransition(i,float(t[i]),before,after,after in seen)
                transitions.append(transition); seen.add(after)
        self.transitions=tuple(transitions); self.transition_times=np.asarray([v.time for v in transitions],dtype=float)
        self.occurrences={i:tuple(run for run in self.runs if run.cluster_id==i) for i in range(k)}
        self.active_motifs=tuple(self.clusters[int(label)].motif for label in self.labels)
        self.motif_returns=tuple(v for v in transitions if v.is_return)
        return self

    def motif_for(self, cluster_id: int) -> Motif:
        """Return the stored motif for a fitted cluster."""
        if not 0<=int(cluster_id)<self.n_clusters: raise IndexError(f'unknown cluster id {cluster_id}')
        return self.clusters[int(cluster_id)].motif

    def replay(self, cluster_id: int, *, variation_strength: float=0., seed: int|None=None, scale_length: int=7) -> Motif:
        """Replay exactly at strength zero, or add bounded scale-safe variation.

        Variations shift by at most one scale degree, move events by at most 0.12
        beats, adjust duration by at most 12%, and alter velocity gently.
        """
        if not np.isfinite(variation_strength) or not 0<=variation_strength<=1: raise ValueError('variation_strength must be in [0,1]')
        if int(scale_length)!=scale_length or scale_length<1: raise ValueError('scale_length must be a positive integer')
        motif=self.motif_for(cluster_id)
        if variation_strength==0: return motif
        rng=np.random.default_rng(self.seed if seed is None else seed); amount=float(variation_strength); result=[]
        for note in motif.notes:
            shift=int(rng.choice([-1,0,1])) if rng.random()<amount else 0
            beat_offset=max(0.,note.beat_offset+float(rng.uniform(-.12,.12))*amount)
            duration=max(.05,note.duration_beats*(1+float(rng.uniform(-.12,.12))*amount))
            velocity=int(np.clip(round(note.velocity+rng.uniform(-7,7)*amount),1,127))
            result.append(MotifNote((note.scale_degree+shift)%int(scale_length),beat_offset,duration,velocity))
        result.sort(key=lambda n:n.beat_offset)
        return Motif(tuple(result),name=f'{motif.name}-variation')

def _default_motif(cluster_id: int, center: np.ndarray, seed: int) -> Motif:
    scale_len=7
    signal=float(np.tanh(np.mean(center))) if center.size else 0.
    base=(int(round((signal+1)*3))+cluster_id)%scale_len
    degrees=[base,(base+2)%scale_len,(base+4)%scale_len,(base+2)%scale_len]
    notes=tuple(MotifNote(degree,offset,duration,velocity) for degree,offset,duration,velocity in zip(degrees,(0.,.5,1.,1.5),(.42,.42,.8,.42),(74,66,78,64)))
    return Motif(notes,name=f'latent-{cluster_id}-seed-{seed}')

def _kmeans(x: np.ndarray,k: int,seed: int,max_iter: int=100) -> tuple[np.ndarray,np.ndarray]:
    if k==1: return x.mean(axis=0,keepdims=True),np.zeros(len(x),dtype=np.int64)
    from sklearn.cluster import KMeans
    model=KMeans(n_clusters=k,random_state=int(seed),n_init=10,max_iter=max_iter,algorithm='lloyd')
    labels=model.fit_predict(x)
    return model.cluster_centers_,labels.astype(np.int64,copy=False)

def _dwell_filter(labels: np.ndarray,timestamps: np.ndarray,min_dwell: float,hysteresis: float) -> np.ndarray:
    if len(labels)<2:return labels.copy()
    result=np.empty_like(labels); active=int(labels[0]); active_since=float(timestamps[0]); candidate=None; candidate_since=active_since
    for i in range(len(labels)):
        raw=int(labels[i]); now=float(timestamps[i])
        if raw==active: candidate=None
        else:
            if candidate!=raw: candidate=raw; candidate_since=now
            if now-active_since>=min_dwell and now-candidate_since>=hysteresis:
                active=raw; active_since=now; candidate=None
        result[i]=active
    return result

def _cluster_runs(labels: np.ndarray,timestamps: np.ndarray) -> tuple[ClusterRun,...]:
    runs=[]; start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            runs.append(ClusterRun(int(labels[start]),start,i-1,float(timestamps[start]),float(timestamps[i-1])))
            start=i
    return tuple(runs)
