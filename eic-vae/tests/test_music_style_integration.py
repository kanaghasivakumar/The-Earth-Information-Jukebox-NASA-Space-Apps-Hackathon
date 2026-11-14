import numpy as np
from music.composer import compose_midi
from music.memory import LatentMemory
from music.styles import available_styles, get_style

def _features(duration=32., step=.1):
    t=np.arange(0,duration,step); n=len(t)
    return {'timestamps':t,'activity':np.ones(n),'movement':np.ones(n),'velocity':np.ones(n),
            'flow_magnitude':np.ones(n),'smoothness':np.ones(n),
            'latent_controls':np.zeros((n,2))}

def test_styles_drive_distinct_composition_scale_progression_rhythm_and_programs(tmp_path):
    features=_features(48.)
    outputs={name:compose_midi(features,tmp_path/f'{name}.mid',style=name,key='D',timing_mode='fixed',target_duration=24.,seed=13)
             for name in available_styles()}
    assert len({(o['tempo'],o['scale'],o['progression'],tuple(o['instrument_programs'].values())) for o in outputs.values()})==5
    for name,result in outputs.items():
        preset=get_style(name)
        assert result['scale']==preset.default_scale
        assert result['tempo']==preset.tempo_bpm
        assert tuple(result['chord_degrees'])==preset.chord_degrees
        assert tuple(result['instrument_programs'][role] for role in ('melody','pad','pluck','bass','drums'))==tuple(i.gm_program for i in preset.instruments)
        assert all(note[1]<=24. for track in result['piano_roll'] for note in track)
        scale={ (2+i)%12 for i in {'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],'dorian':[0,2,3,5,7,9,10],'pentatonic':[0,2,4,7,9]}[result['scale']] }
        assert all(note[2]%12 in scale for track in result['piano_roll'][:4] for note in track)
    assert len(outputs['jazz']['piano_roll'][3])>len(outputs['ambient']['piano_roll'][3])
    assert len(outputs['electronic']['piano_roll'][4])>len(outputs['ambient']['piano_roll'][4])
    motif_rhythms={name:tuple(round(note[0]%(preset.motif.phrase_bars*4*60/preset.tempo_bpm),4) for note in outputs[name]['piano_roll'][0])
                   for name,preset in ((n,get_style(n)) for n in outputs)}
    assert len(set(motif_rhythms.values()))==5
    motif_lengths={name:tuple(round(note[1]-note[0],4) for note in outputs[name]['piano_roll'][0]) for name in outputs}
    assert len(set(motif_lengths.values()))==5

def test_style_none_preserves_legacy_major_defaults(tmp_path):
    result=compose_midi(_features(),tmp_path/'default.mid')
    assert result['tempo']==100 and result['scale']=='major'
    assert result['progression']==(0,4,5,3)
    assert result['instrument_programs']=={'melody':73,'pad':89,'pluck':10,'bass':38,'drums':0}
    beat=60/result['tempo']
    assert all(abs((note[0]/beat)%2)<1e-8 for note in result['piano_roll'][0])

def test_memory_recall_variation_and_phrase_boundary_schedule(tmp_path):
    times=np.arange(0,36,.5)
    latent=np.zeros((len(times),2)); latent[(times>=12)&(times<24),0]=5
    memory=LatentMemory(seed=5).fit(latent,times,n_clusters=2,min_dwell=0,hysteresis=0)
    features=_features(36.,.25)
    exact=compose_midi(features,tmp_path/'exact.mid',style='ambient',timing_mode='fixed',target_duration=36.,memory=memory,variation_strength=0,seed=8)
    replay=compose_midi(features,tmp_path/'replay.mid',style='ambient',timing_mode='fixed',target_duration=36.,memory=memory,variation_strength=0,seed=8)
    varied=compose_midi(features,tmp_path/'varied.mid',style='ambient',timing_mode='fixed',target_duration=36.,memory=memory,variation_strength=.8,seed=8)
    schedule=exact['memory_schedule']; phrase_seconds=4*4*60/get_style('ambient').tempo_bpm
    assert schedule and schedule[0]['start']==0
    assert all(abs(e['start']/phrase_seconds-round(e['start']/phrase_seconds))<1e-8 for e in schedule)
    assert all(e['end']<=e['start']+phrase_seconds+1e-8 for e in schedule)
    assert any(e['is_return'] for e in schedule)
    assert [e['is_return'] for e in schedule]==[False,False,True]
    assert all(not e['is_variation'] for e in schedule)
    assert [e['is_variation'] for e in varied['memory_schedule']]==[False,False,True]
    assert [e['is_return'] for e in varied['memory_schedule']]==[False,False,True]
    assert [e['reuse_count'] for e in varied['memory_schedule']]==[0,0,1]
    assert exact['memory_schedule']==replay['memory_schedule']
    assert (tmp_path/'exact.mid').read_bytes()==(tmp_path/'replay.mid').read_bytes()
    assert [n[2] for n in exact['piano_roll'][0]] != [n[2] for n in varied['piano_roll'][0]]
