import numpy as np
import pytest
from music.memory import LatentMemory, Motif, MotifNote

def test_kmeans_is_deterministic_and_handles_arbitrary_dimension_scales():
    t=np.arange(30,dtype=float)
    z=np.column_stack((np.r_[np.zeros(10),np.ones(10)*8,np.ones(10)*16],np.sin(t)*.001,np.cos(t)*1000))
    a=LatentMemory(seed=7).fit(z,t,n_clusters=3,min_dwell=0,hysteresis=0)
    b=LatentMemory(seed=7).fit(z,t,n_clusters=3,min_dwell=0,hysteresis=0)
    assert a.n_clusters==3 and np.array_equal(a.labels_raw,b.labels_raw)
    assert np.allclose(a.centers,b.centers)
    assert a.labels.shape==(len(t),)

def test_few_or_constant_distinct_points_reduce_cluster_count():
    t=np.array([0.,.2,.4])
    one=LatentMemory().fit(np.ones((3,5)),t,n_clusters=8)
    assert one.n_clusters==1 and np.all(one.labels==0)
    two=LatentMemory().fit(np.array([[0.,0.],[1.,1.],[0.,0.]]),t,n_clusters=9,min_dwell=0,hysteresis=0)
    assert two.n_clusters==2

def test_relative_near_constant_trajectories_collapse_to_one_cluster():
    t=np.arange(20,dtype=float)
    around_large=np.full((20,3),1e6)+np.column_stack((np.linspace(0,1e-3,20),np.sin(t)*2e-4,np.zeros(20)))
    around_zero=np.column_stack((np.linspace(-2e-9,2e-9,20),np.zeros(20)))
    for trajectory in (around_large,around_zero):
        memory=LatentMemory(seed=2).fit(trajectory,t,n_clusters=8)
        assert memory.n_clusters==1
        assert np.unique(memory.labels_raw).tolist()==[0]

def test_timestamp_dwell_hysteresis_suppresses_short_cluster_bounce():
    t=np.arange(9,dtype=float)
    z=np.array([[0.],[0.],[0.],[0.],[5.],[0.],[5.],[5.],[5.]])
    memory=LatentMemory(seed=2).fit(z,t,n_clusters=2,min_dwell=2.,hysteresis=1.)
    assert memory.labels_raw[4]!=memory.labels[4]
    assert memory.labels[4]==memory.labels[3]
    assert memory.labels[7]!=memory.labels[3]
    assert len(memory.transition_times)>=1
    assert memory.runs and sum(run.end_index-run.start_index+1 for run in memory.runs)==len(t)

def test_visualization_fields_expose_runs_transitions_occurrences_and_active_motifs():
    t=np.arange(12,dtype=float); z=np.r_[np.zeros((4,2)),np.ones((4,2))*4,np.zeros((4,2))]
    memory=LatentMemory(seed=3).fit(z,t,n_clusters=2,min_dwell=0,hysteresis=0)
    assert len(memory.active_motifs)==len(t)
    assert len(memory.runs)>=3 and len(memory.transitions)>=2
    assert set(memory.occurrences)=={0,1}
    assert memory.motif_returns and all(v.is_return for v in memory.motif_returns)
    assert all(v.time==t[v.index] for v in memory.transitions)

def test_custom_motif_exact_replay_and_bounded_scale_safe_variation():
    t=np.arange(6,dtype=float); z=np.array([[0.],[0.],[1.],[1.],[0.],[0.]])
    motif=Motif((MotifNote(0,0.,.5,80),MotifNote(2,.5,.75,70),MotifNote(4,1.25,.5,90)),name='remembered')
    memory=LatentMemory(seed=9).fit(z,t,n_clusters=2,min_dwell=0,hysteresis=0,motif_library={0:motif})
    exact=memory.replay(0,variation_strength=0)
    assert exact is motif
    a=memory.replay(0,variation_strength=.7,seed=21,scale_length=7)
    b=memory.replay(0,variation_strength=.7,seed=21,scale_length=7)
    assert a==b and a!=motif
    assert all(0<=n.scale_degree<7 for n in a.notes)
    assert all(abs(n.scale_degree-old.scale_degree)%7 in (0,1,6) for n,old in zip(a.notes,motif.notes))
    assert all(n.velocity in range(1,128) and n.duration_beats>0 for n in a.notes)
    with pytest.raises(ValueError): memory.replay(0,variation_strength=1.1)

def test_high_variation_preserves_shared_motif_transposition():
    t=np.arange(4,dtype=float); z=np.array([[0.],[0.],[1.],[1.]])
    motif=Motif(tuple(MotifNote(degree,offset,.4,80) for degree,offset in zip((0,1,3,5,4,2,6,1,0,3),(0.,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5))),name='long-memory')
    memory=LatentMemory(seed=4).fit(z,t,n_clusters=2,min_dwell=0,hysteresis=0,motif_library={0:motif})
    varied=memory.replay(0,variation_strength=1.,seed=91,scale_length=7,max_transposition_steps=2)
    offsets=[(new.scale_degree-old.scale_degree)%7 for old,new in zip(motif.notes,varied.notes)]
    shared=max(set(offsets),key=offsets.count)
    assert shared in (5,2) # shared transposition is bounded to two degrees
    assert sum(offset!=shared for offset in offsets)<=2 # <=20% local changes
    assert all(0<=note.scale_degree<7 for note in varied.notes)
    assert all(abs(note.beat_offset-old.beat_offset)<=.12+1e-9 for note,old in zip(varied.notes,motif.notes))
    assert all(abs(note.duration_beats-old.duration_beats)<=.12*old.duration_beats+1e-9 for note,old in zip(varied.notes,motif.notes))

def test_timestamp_and_shape_validation():
    m=LatentMemory()
    with pytest.raises(ValueError): m.fit(np.ones((3,2)),[0,0,1])
    with pytest.raises(ValueError): m.fit(np.ones((2,2)),[0],n_clusters=2)
    with pytest.raises(ValueError): m.fit(np.ones((2,2)),[0,1],n_clusters=0)
