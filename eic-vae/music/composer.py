from __future__ import annotations
import math, struct
from pathlib import Path
import numpy as np
from .styles import MusicStyle, get_style
from .memory import LatentMemory, Motif

SCALES={'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],'dorian':[0,2,3,5,7,9,10],'pentatonic':[0,2,4,7,9]}
KEYS={'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

def _quantize(pitch, allowed): return min(allowed,key=lambda p:(abs(p-pitch),p))
def _vlq(n):
    n=max(0,int(n)); b=[n&127]; n>>=7
    while n: b.insert(0,(n&127)|128); n>>=7
    return bytes(b)
def _track(events):
    events.sort(key=lambda e:(e[0],e[1])); data=bytearray(); prev=0
    for tick,_,raw in events: data+=_vlq(tick-prev)+raw; prev=tick
    data+=b'\x00\xff\x2f\x00'; return b'MTrk'+struct.pack('>I',len(data))+data

def _write_fallback(path, tracks, tempo):
    mpqn=round(60_000_000/tempo); chunks=[]
    chunks.append(_track([(0,0,b'\xff\x51\x03'+mpqn.to_bytes(3,'big'))]))
    for channel,program,notes,is_drum in tracks:
        ev=[]
        for start,end,pitch,vel in notes:
            on=round(start*tempo*8); off=round(end*tempo*8); vel=int(vel); pitch=int(pitch)
            ev.extend([(on,1,bytes([0x90|channel,pitch,vel])),(off,0,bytes([0x80|channel,pitch,0]))])
        if not is_drum: ev.append((0,0,bytes([0xC0|channel,int(program)])))
        chunks.append(_track(ev))
    Path(path).write_bytes(b'MThd'+struct.pack('>IHHH',6,1,len(chunks),480)+b''.join(chunks))

def compose_midi(features, output, *, tempo=None, key='C', scale=None, timing_mode='match_video', target_duration=None, weights=None, seed=0, latent_means=None, selected_dims=None, style=None, memory: LatentMemory|None=None, variation_strength=0.25):
    """Compose a repeatable cinematic five-layer score from normalized controls."""
    style_config=get_style(style) if isinstance(style,str) else style
    if style_config is not None and not isinstance(style_config,MusicStyle): raise TypeError('style must be a registered style name or MusicStyle')
    if scale is None: scale=style_config.default_scale if style_config is not None else 'major'
    if tempo is None: tempo=style_config.tempo_bpm if style_config is not None else 100
    if not np.isfinite(variation_strength) or not 0<=variation_strength<=1: raise ValueError('variation_strength must be in [0,1]')
    if memory is not None and (not isinstance(memory,LatentMemory) or not memory.n_clusters): raise ValueError('memory must be a fitted LatentMemory')
    if key not in KEYS or scale not in SCALES: raise ValueError('unsupported key or scale')
    if not np.isfinite(tempo) or tempo<=0: raise ValueError('tempo must be positive and finite')
    if timing_mode not in ('match_video','fixed'): raise ValueError("timing_mode must be 'match_video' or 'fixed'")
    if timing_mode=='fixed' and (target_duration is None or not np.isfinite(target_duration) or target_duration<=0): raise ValueError('fixed timing requires a positive target_duration')
    if timing_mode=='match_video' and target_duration is not None: raise ValueError('target_duration is only valid with fixed timing')
    if weights is None: movement_weight,flow_weight=1.,1.
    else:
        weight_values=np.asarray(weights,dtype=float).reshape(-1)
        if len(weight_values)!=2 or not np.all(np.isfinite(weight_values)) or np.any(weight_values<0) or weight_values.sum()<=0: raise ValueError('weights must be two finite nonnegative values with a positive sum')
        movement_weight,flow_weight=map(float,weight_values)
    t=np.asarray(features['timestamps'],float); a=np.asarray(features['activity'],float)
    controls=np.asarray(features.get('latent_controls',np.zeros((len(t),2))),float)
    if len(t)==0: raise ValueError('cannot compose from empty features')
    def signal_active(name):
        values=np.asarray(features.get(name,[]),dtype=float)
        return bool(values.size and np.nanmax(np.abs(values),initial=0)>1e-6)
    fallback_active=signal_active('activity')
    movement_signal='director_dynamic_intensity' if 'director_dynamic_intensity' in features else 'movement'
    flow_signal='director_percussion_density' if 'director_percussion_density' in features else 'flow_magnitude'
    has_movement=signal_active(movement_signal) if movement_signal in features else fallback_active
    has_flow=signal_active(flow_signal) if flow_signal in features else fallback_active
    melody_signal='director_melodic_activity' if 'director_melodic_activity' in features else movement_signal
    has_melody=signal_active(melody_signal) if melody_signal in features else has_movement
    score_active=(movement_weight>0 and (has_movement or has_melody)) or (flow_weight>0 and has_flow)
    melody_active=movement_weight>0 and has_melody
    source_duration=max(0.,float(t[-1]-t[0])); seconds=float(target_duration if timing_mode=='fixed' else source_duration)
    if seconds<=0: seconds=4*60/tempo
    beat_sec=60/tempo; bar_sec=4*beat_sec
    # Timestamp-derived durations can be a few floating-point microseconds
    # above an exact bar boundary; avoid inventing a trailing empty bar.
    bars=max(1,math.ceil(seconds/bar_sec-1e-6))
    pitch_classes={(KEYS[key]+x)%12 for x in SCALES[scale]}
    allowed=[p for p in range(48,85) if p%12 in pitch_classes]; bass_allowed=[p for p in range(28,53) if p%12 in pitch_classes]
    # Sample at musical eighth-note steps, interpolating timestamp-aware controls.
    def ctl(at,name='activity'):
        director_names={
            'movement':'director_dynamic_intensity',
            'flow_magnitude':'director_percussion_density',
            'velocity':'director_melodic_activity',
            'arpeggio_activity':'director_arpeggio_activity',
        }
        selected=director_names.get(name)
        if selected not in features and name=='arpeggio_activity': selected=director_names['velocity']
        series=np.asarray(features.get(selected,features.get(name,a)),float)
        src_at=t[0]+at/max(seconds,1e-9)*source_duration if timing_mode=='fixed' else min(t[-1],t[0]+at)
        return float(np.interp(src_at,t,series))
    def latent(at,index):
        director_name='director_melody_contour' if index==0 else 'director_harmony_inversion' if index==1 else None
        if director_name and director_name in features:
            series=np.asarray(features[director_name],float)
            src_at=t[0]+at/max(seconds,1e-9)*source_duration if timing_mode=='fixed' else min(t[-1],t[0]+at)
            return float(np.interp(src_at,t,series))
        if controls.ndim!=2 or controls.shape[1]<=index:return 0.
        src_at=t[0]+at/max(seconds,1e-9)*source_duration if timing_mode=='fixed' else min(t[-1],t[0]+at)
        return float(np.interp(src_at,t,controls[:,index]))
    progression=list(style_config.progression) if style_config else ([0,4,5,3] if scale=='major' else [0,5,2,6])
    chord_degrees=tuple(style_config.chord_degrees) if style_config else (0,2,4)
    bars_per_chord=style_config.bars_per_chord if style_config else 2
    phrase_bars=style_config.motif.phrase_bars if style_config else 2
    phrase_degrees=list(style_config.motif.scale_degrees) if style_config else [0,2,4,2]
    style_rhythm=style_config.rhythm if style_config else None
    style_dynamics=style_config.dynamics if style_config else None
    memory_schedule=[]
    if memory is not None:
        seen_clusters=set(); occurrences={}; reuse_counts={}; previous_cluster=None
        phrase_seconds=phrase_bars*bar_sec
        for phrase_index in range(math.ceil(seconds/phrase_seconds)):
            start=phrase_index*phrase_seconds
            source_at=t[0]+start/max(seconds,1e-9)*source_duration if timing_mode=='fixed' else min(t[-1],t[0]+start)
            sample_index=int(np.argmin(np.abs(memory.timestamps-source_at)))
            cluster_id=int(memory.labels[sample_index])
            if cluster_id!=previous_cluster:
                occurrences[cluster_id]=occurrences.get(cluster_id,0)+1
            occurrence=occurrences.get(cluster_id,1)
            is_return=cluster_id in seen_clusters and cluster_id!=previous_cluster
            seen_clusters.add(cluster_id); previous_cluster=cluster_id
            replay_seed=int(seed)+cluster_id*1000003+occurrence*104729
            reuse_count=reuse_counts.get(cluster_id,0); reuse_counts[cluster_id]=reuse_count+1
            applied_variation=variation_strength if is_return else 0.
            motif=memory.replay(cluster_id,variation_strength=applied_variation,seed=replay_seed,scale_length=len(SCALES[scale]))
            actually_varied=bool(applied_variation>0 and motif.notes!=memory.motif_for(cluster_id).notes)
            memory_schedule.append({'start':start,'end':min(start+phrase_seconds,seconds),'cluster_id':cluster_id,
                'occurrence':occurrence,'reuse_count':reuse_count,'is_return':bool(is_return),'is_variation':actually_varied,
                'variation_strength_applied':applied_variation,
                'motif':motif,'motif_name':motif.name,'seed':replay_seed})
    melody=[]; harmony=[]; arpeggio=[]; bass=[]; drums=[]; prev_pitch=60+KEYS[key]
    melody_pitch=[]; note_density=[]; percussion_intensity=[]
    music_times=[]; musical_energy=[]
    previous_voicing=None
    energy_envelope=velocity_envelope=flow_envelope=0.
    for bar in range(bars):
        # Four chords over eight bars, one change every two measures.
        chord=progression[(bar//bars_per_chord)%len(progression)]
        if bar>=bars-2: chord=4 if bar==bars-2 else 0
        scale_intervals=SCALES[scale]; chord_degree=chord%len(scale_intervals)
        chord_degrees_for_bar=[(chord_degree+k)%len(scale_intervals) for k in chord_degrees]
        chord_pcs=[(KEYS[key]+scale_intervals[d])%12 for d in chord_degrees_for_bar]
        latent_voicing=np.clip((latent(bar*bar_sec,1)+1)*.5,0,1)
        inversion=int(round(latent_voicing*2))
        chord_pcs=chord_pcs[inversion:]+chord_pcs[:inversion]
        # Voice-leading: each chord tone stays near its previous voice.
        voicing=[]
        for voice,pc in enumerate(chord_pcs):
            candidates=[p for p in range(48,78) if p%12==pc]
            target=previous_voicing[voice] if previous_voicing else 60+pc
            voicing.append(min(candidates,key=lambda p:abs(p-target)))
        voicing.sort(); previous_voicing=voicing
        if latent_voicing>.7:
            ext_pc=(KEYS[key]+scale_intervals[(chord_degree+6)%len(scale_intervals)])%12
            ext_candidates=[p for p in range(60,84) if p%12==ext_pc]
            if ext_candidates: voicing.append(ext_candidates[0])
        root_pc=(KEYS[key]+scale_intervals[chord_degree])%12
        fifth_degree=(chord_degree+4)%len(scale_intervals); fifth_pc=(KEYS[key]+scale_intervals[fifth_degree])%12
        fraction=(bar+.5)/bars
        if fraction<.12: section='intro'
        elif fraction<.42: section='development'
        elif fraction<.7: section='build'
        elif fraction<.88: section='climax'
        else: section='resolution'
        melodic_on=melody_active and memory is None and style_config is None
        bass_on=True
        pluck_on=True
        section_gain={'intro':.48,'development':.72,'build':.88,'climax':1.,'resolution':.58}[section]
        pluck_gain={'intro':.28,'development':.50,'build':.72,'climax':.86,'resolution':.32}[section]
        bass_gain={'intro':.50,'development':.76,'build':.90,'climax':1.,'resolution':.60}[section]
        melody_gain={'intro':.46,'development':.72,'build':.88,'climax':1.,'resolution':.56}[section]
        if style_dynamics:
            contrast=style_dynamics.section_contrast
            section_gain=(1-contrast)+contrast*section_gain
            pluck_gain=(1-contrast)+contrast*pluck_gain
            bass_gain=(1-contrast)+contrast*bass_gain
            melody_gain=(1-contrast)+contrast*melody_gain
        drum_scale=0.0 if section in ('intro','resolution') else (0.55 if section=='development' else 1.0)
        # A short, recurring, scale-degree motif, reshaped by latent coordinate one.
        phrase_degrees=list(style_config.motif.scale_degrees) if style_config else [0,2,4,2]
        for beat in range(4):
            st=(bar*4+beat)*beat_sec
            if st>=seconds: continue
            energy=ctl(st,'movement'); velocity=ctl(st,'velocity'); flow=ctl(st,'flow_magnitude'); smooth=ctl(st,'smoothness')
            # Hand-authored compatibility for callers that provide activity only.
            if 'movement' not in features: energy=ctl(st)
            if 'velocity' not in features: velocity=ctl(st)
            if 'flow_magnitude' not in features: flow=ctl(st)
            energy=float(np.clip(energy*movement_weight,0,1))
            velocity=float(np.clip(velocity*movement_weight,0,1))
            flow=float(np.clip(flow*flow_weight,0,1))
            if 'smoothness' not in features: smooth=.5
            music_times.append(st); musical_energy.append(energy)
            # Let activity decay over several beats so brief quiet frames do not
            # punch holes in the arrangement. Rhythmic events below are placed
            # on a fixed grid instead of being independently randomized.
            energy=max(energy,energy_envelope*.82)
            velocity=max(velocity,velocity_envelope*.82)
            flow=max(flow,flow_envelope*.78)
            energy_envelope,velocity_envelope,flow_envelope=energy,velocity,flow
            if not score_active:
                note_density.append(0.); melody_pitch.append(np.nan); percussion_intensity.append(0.); continue
            # A continuous harmonic bed anchors every bar, including quiet
            # passages; dynamics still follow the visual activity envelope.
            if beat==0:
                pad_end=min(st+bar_sec,seconds)
                pad_velocity=max(26,int((32+energy*34)*section_gain))
                for pitch in voicing: harmony.append((st,pad_end,pitch,pad_velocity))
            # A steady root/fifth pattern bridges the sustained chords.
            bass_pattern=style_rhythm.bass_beats if style_rhythm else (0.,2.)
            if bass_on and beat in bass_pattern:
                bass_slot=tuple(bass_pattern).index(float(beat))
                pc=root_pc if bass_slot%2==0 else fifth_pc
                bass_pitch=_quantize(36+pc,bass_allowed)
                bass_velocity=max(28,int((38+energy*45)*bass_gain))
                bass_duration=beat_sec*(1.65+.28*np.clip(smooth,0,1))
                bass.append((st,min(st+bass_duration,seconds),bass_pitch,bass_velocity))
            # Keep a regular chord-tone pulse. Section gain changes the texture
            # without allowing random coin flips to create silent gaps.
            if pluck_on and style_rhythm is None:
                arp_index=(bar*4+beat)%len(voicing); pitch=voicing[arp_index]
                arp_velocity=max(24,int((30+velocity*40)*pluck_gain))
                arpeggio.append((st,min(st+beat_sec*.86,seconds),pitch,arp_velocity))
            # Flow controls a stable backbeat's presence and strength.
            drum_start=len(drums)
            if style_rhythm is None and drum_scale and flow>1e-4:
                if beat in (0,2): drums.append((st,min(st+.08,seconds),36,max(24,int((28+energy*64)*flow*drum_scale))))
                if beat in (1,3): drums.append((st,min(st+.08,seconds),38,max(22,int((26+energy*48)*flow*drum_scale))))
                drums.append((st,min(st+.08,seconds),42,max(18,int((20+flow*42)*drum_scale))))
            percussion_intensity.append(float(min(1.,sum(v/127. for _,_,_,v in drums[drum_start:])/2.)))
            motif_density=style_config.motif.activity_density if style_config else 1.
            if melodic_on and beat in (0,2) and velocity>=1.-motif_density:
                phrase_pos=((bar%phrase_bars)*2+beat//2)%len(phrase_degrees); contour=latent(st,0)
                degree=(phrase_degrees[phrase_pos%len(phrase_degrees)]+int(round(contour*(style_config.motif.nearby_variation_steps if style_config else 1.5))))%len(scale_intervals)
                pc=(KEYS[key]+scale_intervals[degree])%12
                desired=_quantize(60+pc,allowed)
                if abs(desired-prev_pitch)>7: desired=prev_pitch+int(np.sign(desired-prev_pitch))*3
                pitch=_quantize(desired,allowed); prev_pitch=pitch
                if style_dynamics:
                    crescendo=np.clip((fraction-style_dynamics.crescendo_start)/max(style_dynamics.crescendo_end-style_dynamics.crescendo_start,1e-9),0,1)
                    level=np.clip((energy*style_dynamics.accent_gain+(1-energy)*crescendo)*melody_gain,0,1)
                    melody_velocity=int(style_dynamics.floor_velocity+(style_dynamics.peak_velocity-style_dynamics.floor_velocity)*level)
                else: melody_velocity=max(26,int((32+energy*50)*melody_gain))
                melody_duration=beat_sec*(1.65+.25*np.clip(smooth,0,1))
                melody.append((st,min(st+melody_duration,seconds),pitch,melody_velocity))
                melody_pitch.append(float(pitch))
            else: melody_pitch.append(np.nan)
            note_density.append(float(sum(1 for n in melody if n[0]==st)))
    if style_rhythm and score_active:
        for bar in range(bars):
            fraction=(bar+.5)/bars
            if fraction<.12: section='intro'
            elif fraction<.42: section='development'
            elif fraction<.7: section='build'
            elif fraction<.88: section='climax'
            else: section='resolution'
            drum_scale=0. if section in ('intro','resolution') else .55 if section=='development' else 1.
            if pluck_on:
                pad_at=bar*bar_sec
                chord_tones=[n[2] for n in harmony if abs(n[0]-pad_at)<1e-9]
                if chord_tones:
                    for arp_i,beatpos in enumerate(np.arange(0.,4.,style_rhythm.arpeggio_subdivision)):
                        st=bar*bar_sec+float(beatpos)*beat_sec
                        if st>=seconds: continue
                        velocity=float(np.clip(ctl(st,'arpeggio_activity')*movement_weight,0,1)) if 'velocity' in features or 'director_arpeggio_activity' in features else float(np.clip(ctl(st)*movement_weight,0,1))
                        threshold=1.-style_config.motif.activity_density
                        if velocity<threshold: continue
                        section_arp={'intro':.28,'development':.5,'build':.72,'climax':.86,'resolution':.32}[section]
                        arp_gain=(1-style_dynamics.section_contrast)+style_dynamics.section_contrast*section_arp
                        note_velocity=int(style_dynamics.floor_velocity+(style_dynamics.peak_velocity-style_dynamics.floor_velocity)*np.clip(velocity*arp_gain*style_dynamics.accent_gain,0,1))
                        pitch=chord_tones[arp_i%len(chord_tones)]
                        arpeggio.append((st,min(st+style_rhythm.arpeggio_subdivision*beat_sec*.84,seconds),pitch,note_velocity))
            if drum_scale<=0: continue
            for beatpos in style_rhythm.kick_beats+style_rhythm.snare_beats:
                st=bar*bar_sec+beatpos*beat_sec
                if st>=seconds: continue
                flow=float(np.clip(ctl(st,'flow_magnitude')*flow_weight,0,1)) if 'flow_magnitude' in features else float(np.clip(ctl(st)*flow_weight,0,1))
                energy=float(np.clip(ctl(st,'movement')*movement_weight,0,1)) if 'movement' in features else float(np.clip(ctl(st)*movement_weight,0,1))
                if flow<=1e-4: continue
                is_kick=beatpos in style_rhythm.kick_beats; is_snare=beatpos in style_rhythm.snare_beats
                if is_kick: drums.append((st,min(st+.08,seconds),36,max(22,int((28+energy*64)*flow*drum_scale))))
                if is_snare: drums.append((st,min(st+.08,seconds),38,max(20,int((25+energy*52)*flow*drum_scale))))
            hat_step=style_rhythm.hat_subdivision
            for step in np.arange(0.,4.,hat_step):
                st=bar*bar_sec+float(step)*beat_sec
                if st>=seconds: continue
                flow=float(np.clip(ctl(st,'flow_magnitude')*flow_weight,0,1)) if 'flow_magnitude' in features else float(np.clip(ctl(st)*flow_weight,0,1))
                if flow>1e-4: drums.append((st,min(st+.06,seconds),42,max(16,int((18+flow*40)*drum_scale))))
    if style_config is not None and memory is None and melody_active:
        style_motif=style_config.motif; phrase_seconds=phrase_bars*bar_sec
        for phrase_index in range(math.ceil(seconds/phrase_seconds)):
            phrase_start=phrase_index*phrase_seconds; phrase_end=min(phrase_start+phrase_seconds,seconds)
            for note_index,scale_degree in enumerate(style_motif.scale_degrees):
                beat_offset=note_index*(phrase_bars*4/len(style_motif.scale_degrees))
                # Syncopation delays alternating motif attacks by a small fraction of a beat.
                if note_index%2: beat_offset+=style_rhythm.syncopation*.25
                st=phrase_start+beat_offset*beat_sec
                if st>=phrase_end: continue
                velocity=float(np.clip(ctl(st,'velocity')*movement_weight,0,1)) if 'velocity' in features else float(np.clip(ctl(st)*movement_weight,0,1))
                if velocity<1.-style_motif.activity_density: continue
                contour=latent(st,0); degree=(scale_degree+int(round(contour*style_motif.nearby_variation_steps)))%len(scale_intervals)
                pc=(KEYS[key]+scale_intervals[degree])%12; pitch=_quantize(60+pc,allowed)
                section_fraction=(st/bar_sec+.5)/bars
                if section_fraction<.12: section='intro'
                elif section_fraction<.42: section='development'
                elif section_fraction<.7: section='build'
                elif section_fraction<.88: section='climax'
                else: section='resolution'
                section_level={'intro':.46,'development':.72,'build':.88,'climax':1.,'resolution':.56}[section]
                section_gain=(1-style_dynamics.section_contrast)+style_dynamics.section_contrast*section_level
                energy=float(np.clip(ctl(st,'movement')*movement_weight,0,1)) if 'movement' in features else float(np.clip(ctl(st)*movement_weight,0,1))
                crescendo=np.clip((section_fraction-style_dynamics.crescendo_start)/max(style_dynamics.crescendo_end-style_dynamics.crescendo_start,1e-9),0,1)
                level=np.clip((energy*style_dynamics.accent_gain+(1-energy)*crescendo)*section_gain,0,1)
                note_velocity=int(style_dynamics.floor_velocity+(style_dynamics.peak_velocity-style_dynamics.floor_velocity)*level)
                note_spacing=(phrase_bars*4/len(style_motif.scale_degrees))*beat_sec
                duration=min(max(.08,note_spacing*style_motif.legato),phrase_end-st,seconds-st)
                melody.append((st,st+duration,pitch,note_velocity))
    if memory is not None and melody_active:
        for event in memory_schedule:
            phrase_start=float(event['start']); phrase_end=float(event['end'])
            for note in event['motif'].notes:
                st=phrase_start+note.beat_offset*beat_sec
                if st>=phrase_end or st>=seconds: continue
                end=min(st+note.duration_beats*beat_sec,phrase_end,seconds)
                degree=note.scale_degree%len(scale_intervals)
                if 'director_melody_contour' in features:
                    contour_steps=style_config.motif.nearby_variation_steps if style_config else 1
                    degree=(degree+int(round(latent(st,0)*contour_steps)))%len(scale_intervals)
                pc=(KEYS[key]+scale_intervals[degree])%12
                pitch=_quantize(60+pc,allowed); energy=float(np.clip(ctl(st,'movement')*movement_weight,0,1)) if 'movement' in features else float(np.clip(ctl(st)*movement_weight,0,1))
                velocity=note.velocity if variation_strength==0 else int(np.clip(note.velocity*(.55+.45*energy),1,127))
                if 'director_melodic_activity' in features:
                    activity=float(np.clip(ctl(st,'velocity')*movement_weight,0,1))
                    threshold=1.-(style_config.motif.activity_density if style_config else 1.)
                    if activity<threshold: continue
                    velocity=int(np.clip(velocity*(.35+.65*activity),1,127))
                melody.append((st,end,pitch,velocity))
    melody.sort(key=lambda n:n[0]); harmony.sort(key=lambda n:n[0]); arpeggio.sort(key=lambda n:n[0]); bass.sort(key=lambda n:n[0]); drums.sort(key=lambda n:n[0])
    default_programs={'melody':73,'pad':89,'pluck':10,'bass':38,'drums':0}
    if style_config:
        default_programs.update({item.role:item.gm_program for item in style_config.instruments})
        instrument_specs=tuple(style_config.instruments)
    else:
        from .styles import InstrumentSpec
        instrument_specs=tuple(InstrumentSpec(role,program,timbre) for role,program,timbre in (
            ('melody',73,'soft flute'),('pad',89,'warm pad'),('pluck',10,'music box'),('bass',38,'rounded synth bass'),('drums',0,'soft percussion')))
    instrument_programs=dict(default_programs)
    tracks=[(0,default_programs['melody'],melody,False),(1,default_programs['pad'],harmony,False),
            (2,default_programs['pluck'],arpeggio,False),(3,default_programs['bass'],bass,False),(9,0,drums,True)]
    try:
        import pretty_midi
        pm=pretty_midi.PrettyMIDI(initial_tempo=tempo,resolution=480)
        for name,channel,program,notes,is_drum in [('Melody',0,default_programs['melody'],melody,False),('Pad',1,default_programs['pad'],harmony,False),('Arpeggio',2,default_programs['pluck'],arpeggio,False),('Bass',3,default_programs['bass'],bass,False),('Drums',9,0,drums,True)]:
            inst=pretty_midi.Instrument(program=program,is_drum=is_drum,name=name)
            for st,en,p,v in notes: inst.notes.append(pretty_midi.Note(velocity=v,pitch=p,start=st,end=en))
            pm.instruments.append(inst)
        pm.write(str(output))
    except ImportError: _write_fallback(output,tracks,tempo)
    density=[]; pitches=[]; percussion=[]
    for st in music_times:
        frame_melody=[n for n in melody if st<=n[0]<st+beat_sec]
        frame_drums=[n for n in drums if st<=n[0]<st+beat_sec]
        density.append(float(len(frame_melody))); pitches.append(float(frame_melody[0][2]) if frame_melody else np.nan)
        percussion.append(float(min(1.,sum(n[3]/127. for n in frame_drums)/2.)))
    musical_features={'timestamps':np.asarray(music_times),'musical_energy':np.asarray(musical_energy),'note_density':np.asarray(density),'melody_pitch':np.asarray(pitches),'percussion_intensity':np.asarray(percussion)}
    schedule=[]
    for event in memory_schedule:
        public={key:value for key,value in event.items() if key!='motif'}
        public['motif_notes']=[{'scale_degree':n.scale_degree,'beat_offset':n.beat_offset,'duration_beats':n.duration_beats,'velocity':n.velocity} for n in event['motif'].notes]
        schedule.append(public)
    return {'duration':seconds,'tempo':tempo,'scale':scale,'progression':tuple(progression),'chord_degrees':chord_degrees,
        'bars':bars,'selected_dims':selected_dims,'piano_roll':[melody,harmony,arpeggio,bass,drums],
        'musical_features':musical_features,'memory_schedule':schedule,'instrument_specs':instrument_specs,
        'instrument_programs':instrument_programs}
