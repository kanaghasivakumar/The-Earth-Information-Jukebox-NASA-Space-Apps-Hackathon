"""Chunked deterministic WAV rendering with a light built-in fallback synth."""
from __future__ import annotations
import math, tempfile, wave
from pathlib import Path
import numpy as np

def render_basic_wav(piano_roll, duration, output, *, sample_rate=44100, seed=0, instrument_specs=None):
    """Render score groups to normalized stereo PCM without full-track buffers."""
    if duration<=0 or sample_rate<=0: raise ValueError('duration and sample_rate must be positive')
    total=int(round(duration*sample_rate)); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    notes=[]
    for group_ix,group in enumerate(piano_roll):
        for note_ix,(start,end,pitch,velocity) in enumerate(group):
            notes.append((group_ix,note_ix,float(start),float(end),int(pitch),int(velocity)))
    role_order=('melody','pad','pluck','bass','drums')
    timbres={}
    for spec in instrument_specs or ():
        role=spec.get('role') if isinstance(spec,dict) else getattr(spec,'role',None)
        timbre=spec.get('timbre','') if isinstance(spec,dict) else getattr(spec,'timbre','')
        if role is not None: timbres[str(role)]=str(timbre).lower()
    chunk_size=sample_rate
    peak=0.
    with tempfile.NamedTemporaryFile(prefix='earth-jukebox-',suffix='.f32',dir=output.parent,delete=False) as temp:
        tmp=Path(temp.name)
        try:
            for first in range(0,total,chunk_size):
                count=min(chunk_size,total-first); t=(first+np.arange(count,dtype=np.float32))/sample_rate
                mix=np.zeros((count,2),dtype=np.float32)
                for group_ix,note_ix,start,end,pitch,velocity in notes:
                    if end<=start or end<=first/sample_rate or start>=(first+count)/sample_rate: continue
                    lo=max(0,int(math.floor(start*sample_rate))-first); hi=min(count,int(math.ceil(end*sample_rate))-first)
                    if hi<=lo: continue
                    ix=np.arange(first+lo,first+hi,dtype=np.float64); local=ix/sample_rate-start; dur=max(end-start,1e-5)
                    amp=velocity/127*.16
                    attack=np.minimum(1.,local/.008); release=np.minimum(1.,np.maximum(0.,(end-ix/sample_rate)/.025))
                    env=attack*release
                    if group_ix==4:
                        if pitch==36: # kick: descending sine sweep
                            phase=2*np.pi*(55*local-18*local*local); sig=np.sin(phase)*np.exp(-local*24)
                        else: # snare and closed hat: deterministic short noise burst
                            rng=np.random.default_rng(seed+note_ix*104729+pitch*97)
                            noise=rng.standard_normal(max(1,int(math.ceil(dur*sample_rate))))
                            relix=np.clip(np.arange(first+lo,first+hi)-int(round(start*sample_rate)),0,len(noise)-1)
                            sig=noise[relix]*np.exp(-local*(34 if pitch==38 else 75))
                            if pitch==38: sig=.65*sig+.35*np.sin(2*np.pi*180*local)*np.exp(-local*20)
                    elif group_ix==0:
                        hz=440.*(2.**((pitch-69)/12.)); phase=2*np.pi*hz*local
                        timbre=timbres.get(role_order[group_ix],'')
                        if any(word in timbre for word in ('violin','sax','reed','saw')):
                            sig=sum(np.sin(k*phase)/k for k in range(1,7))
                            sig/=sum(1/k for k in range(1,7))
                        elif any(word in timbre for word in ('piano','electric')):
                            sig=.68*np.sin(phase)+.24*np.sin(2*phase)+.08*np.sin(3*phase)
                        else: # flute-like, mostly fundamental
                            sig=.88*np.sin(phase)+.12*np.sin(2*phase)
                    elif group_ix==1: # warm, slowly swelling pad
                        hz=440.*(2.**((pitch-69)/12.)); phase=2*np.pi*hz*local
                        timbre=timbres.get(role_order[group_ix],'')
                        if 'string' in timbre or 'violin' in timbre:
                            sig=.4*np.sin(phase)+.26*np.sin(1.006*phase)+.22*np.sin(2*phase)+.12*np.sin(3*phase)
                        elif any(word in timbre for word in ('poly','synth')):
                            sig=.42*np.sin(phase)+.3*np.sin(1.012*phase)+.18*np.sin(2*phase)+.1*np.sin(3*phase)
                        else:
                            sig=.55*np.sin(phase)+.3*np.sin(1.006*phase)+.15*np.sin(2*phase)
                        env*=np.minimum(1.,local/.18)
                    elif group_ix==2: # bright plucked music-box arpeggio
                        hz=440.*(2.**((pitch-69)/12.)); phase=2*np.pi*hz*local
                        timbre=timbres.get(role_order[group_ix],'')
                        if 'piano' in timbre:
                            sig=.72*np.sin(phase)+.2*np.sin(2*phase)+.08*np.sin(3*phase)
                        elif any(word in timbre for word in ('vibraphone','glock','bell')):
                            sig=.55*np.sin(phase)+.3*np.sin(2.76*phase)+.15*np.sin(5.4*phase)
                        else:
                            sig=.65*np.sin(phase)+.25*np.sin(2.7*phase)+.1*np.sin(4.1*phase)
                        env*=np.exp(-local*7)
                    else: # rounded bass
                        hz=440.*(2.**((pitch-69)/12.)); phase=2*np.pi*hz*local
                        timbre=timbres.get(role_order[group_ix],'')
                        if any(word in timbre for word in ('acoustic','contrabass','cello')):
                            sig=.8*np.sin(phase)+.2*np.sin(2*phase)
                        else:
                            sig=2/np.pi*np.arcsin(np.sin(phase))
                    voice=(amp*env*sig).astype(np.float32)
                    # Slight track panning, centered drums.
                    pan=(-.25 if group_ix==0 else .25 if group_ix==1 else 0.)
                    mix[lo:hi,0]+=voice*(1-pan); mix[lo:hi,1]+=voice*(1+pan)
                peak=max(peak,float(np.max(np.abs(mix),initial=0)))
                mix.tofile(temp)
        finally: pass
    # Two-pass peak normalization keeps working memory bounded to one second.
    gain=min(8.,.95/max(peak,1e-8))
    with wave.open(str(output),'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(sample_rate)
        with tmp.open('rb') as src:
            remaining=total
            while remaining:
                frames=min(chunk_size,remaining); data=np.fromfile(src,dtype=np.float32,count=frames*2).reshape(-1,2)
                pcm=np.clip(data*gain,-.98,.98); wav.writeframesraw(np.rint(pcm*32767).astype('<i2').tobytes()); remaining-=frames
    tmp.unlink(missing_ok=True)
    return {'sample_rate':sample_rate,'channels':2,'frames':total,'peak':min(.98,peak*gain),'renderer':'basic'}
