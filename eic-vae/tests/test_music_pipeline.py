import struct
import numpy as np
import pytest
from music.features import extract_features, normalize_percentile
from music.composer import compose_midi
from music.render import render_basic_wav
from music.plots import save_plots
import wave

def _vlq(data,pos):
    value=0
    while True:
        byte=data[pos]; pos+=1; value=(value<<7)|(byte&127)
        if byte<128:return value,pos

def _midi_event_times(raw):
    ticks=[]; pos=14
    for _ in range(struct.unpack('>H',raw[10:12])[0]):
        size=struct.unpack('>I',raw[pos+4:pos+8])[0]; data=raw[pos+8:pos+8+size]; pos+=8+size
        i=0; tick=0; running_status=None
        while i<len(data):
            delta,i=_vlq(data,i); tick+=delta; ticks.append(tick)
            status=data[i]
            if status & 0x80:
                i+=1
                if status < 0xf0: running_status=status
            elif running_status is not None:
                status=running_status
            else:
                raise AssertionError('MIDI running status used before a channel status')
            if status==0xff:
                meta=data[i]; i+=1; length,i=_vlq(data,i)
                if meta==0x2f:break
                i+=length
            elif status&0xf0 in (0x80,0x90): i+=2
            elif status&0xf0==0xC0: i+=1
            else: raise AssertionError(f'unexpected MIDI status {status:#x}')
    return ticks

def test_static_safety_and_percentile_normalization():
    assert np.all(normalize_percentile([0,0,0])==0)
    assert np.allclose(normalize_percentile([0,1,2]),[0,.5,1])
    f=extract_features([0,1,2],np.ones((3,2)))
    assert np.all(f['activity']==0)

def test_timestamp_aware_features_and_validation():
    z=np.array([[0.],[0.1],[1.],[4.]])
    a=extract_features([0,1,2,3],z)
    b=extract_features([0,1,4,9],z)
    assert np.max(a['movement']) > 0
    # normalized controls preserve shape while raw per-second velocity changes.
    assert not np.array_equal(a['velocity'],b['velocity'])
    with pytest.raises(ValueError): extract_features([0,0,1],z)

def test_midi_valid_tracks_duration_chords_and_repeatability(tmp_path):
    t=np.arange(0,4,.2); f={'timestamps':t,'activity':np.linspace(0,1,len(t))}
    p1=tmp_path/'a.mid'; p2=tmp_path/'b.mid'
    res=compose_midi(f,p1,tempo=120,key='C',scale='major',timing_mode='fixed',target_duration=8,seed=22)
    compose_midi(f,p2,tempo=120,key='C',scale='major',timing_mode='fixed',target_duration=8,seed=22)
    raw=p1.read_bytes(); assert raw[:4]==b'MThd'
    assert struct.unpack('>H',raw[12:14])[0]==480
    assert raw.count(b'MTrk')==6
    assert b'\xc0\x49' in raw and b'\xc1\x59' in raw and b'\xc2\x0a' in raw and b'\xc3\x26' in raw
    assert raw==p2.read_bytes()
    assert res['duration']==8 and res['bars']==4

def test_composer_quantizes_to_scale_and_uses_selected_dims(tmp_path):
    f={'timestamps':np.arange(10.),'activity':np.ones(10)}
    info=compose_midi(f,tmp_path/'x.mid',key='D',scale='minor',timing_mode='fixed',target_duration=4,selected_dims=[2,0])
    assert info['selected_dims']==[2,0]

def test_latent_dimensions_control_melody_and_harmony(tmp_path):
    t=np.arange(32.)*.25; activity=np.ones(len(t))
    def make(c0,c1,name):
        f={'timestamps':t,'activity':activity,'latent_controls':np.column_stack((np.full(len(t),c0),np.full(len(t),c1)))}
        return compose_midi(f,tmp_path/name,tempo=100,key='D',scale='minor',timing_mode='fixed',target_duration=7.3,seed=9)['piano_roll']
    neutral=make(-.8,-.8,'n.mid'); changed_melody=make(.8,-.8,'m.mid'); changed_harmony=make(-.8,.8,'h.mid')
    assert [n[2] for n in neutral[0]] != [n[2] for n in changed_melody[0]]
    assert [n[2] for n in neutral[1]] != [n[2] for n in changed_harmony[1]]
    d_minor={(2+x)%12 for x in [0,2,3,5,7,8,10]}
    assert all(n[2]%12 in d_minor for track in neutral[:4] for n in track)

def test_static_activity_suppresses_events_and_target_duration_clips(tmp_path):
    t=np.arange(20.)*.2
    static=extract_features(t,np.zeros((len(t),2)),flow_magnitude=np.zeros(len(t)))
    quiet=compose_midi(static,tmp_path/'quiet.mid',tempo=90,timing_mode='fixed',target_duration=3.7,seed=2)
    assert all(len(track)==0 for track in quiet['piano_roll'])
    active={'timestamps':t,'activity':np.ones(len(t)),'latent_controls':np.zeros((len(t),2))}
    result=compose_midi(active,tmp_path/'clip.mid',tempo=90,timing_mode='fixed',target_duration=3.7,seed=2)
    assert all(note[1] <= 3.7 for track in result['piano_roll'] for note in track)
    assert all(event[0] < 3.7 for event in result['piano_roll'][0])
    assert len(result['piano_roll'][4])>0
    # PrettyMIDI rounds timestamps to integer ticks; allow one tick at the clip boundary.
    assert max(_midi_event_times((tmp_path/'clip.mid').read_bytes())) <= round(3.7*90*8)+1

def test_fallback_uses_tempo_for_tick_conversion(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'pretty_midi',None)
    t=np.arange(8.)*.25; f={'timestamps':t,'activity':np.ones(len(t)),'latent_controls':np.zeros((len(t),2))}
    path=tmp_path/'tempo90.mid'; compose_midi(f,path,tempo=90,timing_mode='fixed',target_duration=2,seed=1)
    raw=path.read_bytes(); assert struct.unpack('>H',raw[12:14])[0]==480
    # At 90 BPM one second is 720 ticks; activity note onsets are on beat boundaries.
    assert raw.count(b'MTrk')==6 and b'\xff\x51\x03\x0a\x2c\x2b' in raw
    starts=[]; pos=14
    for _ in range(6):
        size=struct.unpack('>I',raw[pos+4:pos+8])[0]; data=raw[pos+8:pos+8+size]; pos+=8+size
        i=0; tick=0
        while i<len(data):
            delta,i=_vlq(data,i); tick+=delta
            status=data[i]; i+=1
            if status==0xff:
                meta=data[i]; i+=1; length,i=_vlq(data,i)
                if meta==0x2f:break
                i+=length
            elif status&0xf0 in (0x80,0x90):
                pitch,velocity=data[i:i+2]; i+=2
                if status&0xf0==0x90 and velocity:starts.append(tick)
            elif status&0xf0==0xC0: i+=1
            else: raise AssertionError(f'unexpected MIDI status {status:#x}')
    assert 480 in starts and 960 in starts

def test_timing_modes_and_input_validation(tmp_path):
    f={'timestamps':np.arange(10.),'activity':np.ones(10)}
    matched=compose_midi(f,tmp_path/'match.mid',tempo=100)
    assert matched['duration']==9
    near_bar={'timestamps':np.array([0.,8.0000002]),'activity':np.ones(2)}
    assert compose_midi(near_bar,tmp_path/'near_bar.mid',tempo=120)['bars']==4
    with pytest.raises(ValueError): compose_midi(f,tmp_path/'missing.mid',timing_mode='fixed')
    with pytest.raises(ValueError): compose_midi(f,tmp_path/'wrong.mid',target_duration=4)
    with pytest.raises(ValueError): compose_midi(f,tmp_path/'tempo.mid',tempo=0)
    with pytest.raises(ValueError): compose_midi(f,tmp_path/'weights.mid',weights=[-1,1])
    with pytest.raises(ValueError): extract_features(np.arange(2),np.zeros((2,1)),smoothing=0)

def test_movement_and_flow_weights_change_score(tmp_path):
    t=np.arange(40.)*.1
    f={'timestamps':t,'activity':np.ones(len(t)),'movement':np.ones(len(t)),'velocity':np.ones(len(t)),'flow_magnitude':np.ones(len(t)),'smoothness':np.ones(len(t)),'latent_controls':np.zeros((len(t),2))}
    full=compose_midi(f,tmp_path/'full.mid',tempo=120,weights=[1,1],seed=3)
    no_movement=compose_midi(f,tmp_path/'still.mid',tempo=120,weights=[0,1],seed=3)
    no_flow=compose_midi(f,tmp_path/'dry.mid',tempo=120,weights=[1,0],seed=3)
    assert sum(full['musical_features']['musical_energy'])>sum(no_movement['musical_features']['musical_energy'])
    assert len(full['piano_roll'][0])>len(no_movement['piano_roll'][0])
    assert len(full['piano_roll'][4])>len(no_flow['piano_roll'][4])

def test_ambient_five_layer_arrangement_and_voice_led_progression(tmp_path):
    t=np.arange(0,64,.2); n=len(t)
    f={'timestamps':t,'activity':np.ones(n),'movement':np.ones(n),'velocity':np.ones(n),'flow_magnitude':np.ones(n),'smoothness':np.ones(n),'latent_controls':np.zeros((n,2))}
    path=tmp_path/'ambient.mid'; result=compose_midi(f,path,tempo=120,key='C',scale='major',seed=4)
    melody,pad,pluck,bass,drums=result['piano_roll']
    assert all(len(track)>0 for track in (melody,pad,pluck,bass,drums))
    assert result['musical_features']['percussion_intensity'].max()<=1
    assert result['musical_features']['percussion_intensity'].max()>0
    # The opening follows I-V-vi-IV across two-bar chord phrases; each voice remains in scale.
    beat=0.5; bar=4*beat
    pad_roots=[]
    for start in (0,2*bar,4*bar,6*bar):
        pcs={note[2]%12 for note in pad if abs(note[0]-start)<1e-8}
        pad_roots.append(pcs)
    expected=[{0,4,7},{7,11,2},{9,0,4},{5,9,0}]
    assert all(want.issubset(got) for want,got in zip(expected,pad_roots))
    major={0,2,4,5,7,9,11}
    assert all(note[2]%12 in major for track in (melody,pad,pluck,bass) for note in track)
    assert max(note[1] for note in pad if note[0]<bar)<=bar+1e-9
    assert max(note[1] for note in pad if note[0]<bar)>=bar-1e-9

def test_active_arrangement_has_no_random_pauses(tmp_path):
    t=np.arange(0,16,.2); n=len(t)
    movement=np.zeros(n); movement[[0,n-1]]=1
    flow=np.zeros(n); flow[[0,n-1]]=1
    f={'timestamps':t,'activity':np.maximum(movement,flow),'movement':movement,'velocity':movement,
       'flow_magnitude':flow,'smoothness':np.ones(n),'latent_controls':np.zeros((n,2))}
    first=compose_midi(f,tmp_path/'continuous_a.mid',tempo=120,seed=1)
    second=compose_midi(f,tmp_path/'continuous_b.mid',tempo=120,seed=99)
    melody,pad,pluck,bass,drums=first['piano_roll']
    assert first['bars']==8
    assert len(pad)==first['bars']*3
    assert len(bass)==first['bars']*2
    assert len(pluck)==first['bars']*4
    assert len(melody)==first['bars']*2
    assert [note[0] for note in melody]==[note[0] for note in second['piano_roll'][0]]
    assert [note[0] for note in pad]==[note[0] for note in second['piano_roll'][1]]
    # The harmonic bed sustains through every bar, even when local controls dip.
    assert all(any(note[0]<=bar*2 and note[1]>=bar*2+1.99 for note in pad)
               for bar in range(first['bars']-1))

def test_basic_wav_is_stereo_44100_finite_and_deterministic(tmp_path):
    score=[[(0,.18,69,100)],[(0,.2,60,75)],[(0,.16,64,75)],[(0,.2,36,90)],[(.05,.13,36,110),(.15,.22,42,65)]]
    a=tmp_path/'a.wav'; b=tmp_path/'b.wav'
    info=render_basic_wav(score,.25,a,sample_rate=44100,seed=11)
    render_basic_wav(score,.25,b,sample_rate=44100,seed=11)
    with wave.open(str(a),'rb') as wav:
        assert wav.getframerate()==44100 and wav.getnchannels()==2
        assert wav.getnframes()==round(.25*44100)
        samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')
    assert np.isfinite(samples).all() and np.max(np.abs(samples))/32767<=1
    assert a.read_bytes()==b.read_bytes() and info['peak']<=1

def test_static_score_plots_write_feature_and_empty_piano_roll(tmp_path):
    t=np.arange(5.)
    features=extract_features(t,np.zeros((len(t),2)),flow_magnitude=np.zeros(len(t)))
    score=compose_midi(features,tmp_path/'static.mid',timing_mode='match_video')
    save_plots(features,np.zeros((len(t),2)),tmp_path,score['piano_roll'],score['musical_features'])
    assert (tmp_path/'earth_jukebox_features.png').is_file()
    assert (tmp_path/'earth_jukebox_piano_roll.png').is_file()


def test_render_failure_leaves_no_scratch_file(tmp_path, monkeypatch):
    import music.render as render
    def boom(*a, **k): raise OSError('disk full')
    monkeypatch.setattr(render.wave, 'open', boom)
    score=[[(0,.18,69,100)],[(0,.2,60,75)],[(0,.16,64,75)],[(0,.2,36,90)],[(.05,.13,36,110)]]
    with pytest.raises(OSError):
        render_basic_wav(score,.25,tmp_path/'a.wav',sample_rate=44100,seed=11)
    assert list(tmp_path.glob('earth-jukebox-*.f32'))==[]
