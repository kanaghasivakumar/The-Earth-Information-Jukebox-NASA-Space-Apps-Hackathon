"""Small dependency-light feature and piano-roll plots (Pillow only)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

COLORS=[(30,110,190),(220,90,50),(60,150,90),(145,80,170),(220,160,35),(40,155,165),(185,70,110),(100,100,100)]

def _axes(draw, box, title, xlab, ylab):
    x0,y0,x1,y1=box; draw.rectangle(box,outline=(120,120,120),width=1)
    draw.text((x0,y0-24),title,fill=(20,20,20)); draw.text((x0+360,y1+7),xlab,fill=(20,20,20)); draw.text((5,y0+4),ylab,fill=(20,20,20))

def _line(draw, box, xs, ys, color, yrange=None, max_points=220):
    x0,y0,x1,y1=box; xs=np.asarray(xs,float); ys=np.asarray(ys,float)
    if len(xs)<2:return
    valid=np.isfinite(xs)&np.isfinite(ys); xs=xs[valid]; ys=ys[valid]
    if len(xs)<2:return
    if len(ys)>5:
        ys=np.convolve(np.pad(ys,(2,2),mode='edge'),np.ones(5)/5,mode='valid')
    if len(xs)>max_points:
        picks=np.unique(np.linspace(0,len(xs)-1,max_points).astype(int)); xs=xs[picks]; ys=ys[picks]
    lo,hi=yrange if yrange else (float(np.nanmin(ys)),float(np.nanmax(ys)))
    if hi-lo<1e-12: hi=lo+1
    xx=x0+(xs-xs.min())/max(float(xs.max()-xs.min()),1e-12)*(x1-x0)
    yy=y1-(ys-lo)/(hi-lo)*(y1-y0)
    draw.line(list(zip(xx.astype(int),yy.astype(int))),fill=color,width=2)

def save_plots(features, means, out_dir, midi=None, musical_features=None, *, timing_mode='match_video', duration=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); t=np.asarray(features['timestamps'],float)
    source_duration=max(float(t[-1]-t[0]),1e-9) if len(t)>1 else 1.
    plot_t=(t-t[0])*(float(duration)/source_duration) if timing_mode=='fixed' and duration else (t-t[0])
    xlabel='Musical time (seconds)' if timing_mode=='fixed' else 'Video/musical time (seconds)'
    image=Image.new('RGB',(1200,1000),'white'); draw=ImageDraw.Draw(image)
    top=(105,55,1160,255); bot=(105,365,1160,535); music_box=(105,650,1160,875)
    _axes(draw,top,'VAE latent patterns',xlabel,'Pattern value')
    for i in range(means.shape[1]):
        _line(draw,top,plot_t,means[:,i],COLORS[i%len(COLORS)])
        draw.text((110+(i%8)*125,275+(i//8)*17),f'latent {i}',fill=COLORS[i%len(COLORS)])
    _axes(draw,bot,'Motion and activity',xlabel,'Normalized value')
    for i,k in enumerate(('movement','velocity','flow_magnitude','activity')):
        _line(draw,bot,plot_t,features[k],COLORS[i]); draw.text((110+i*200,550),k,fill=COLORS[i])
    if musical_features:
        mt=np.asarray(musical_features['timestamps'])
        _axes(draw,music_box,'Musical controls aligned to musical seconds','Musical time (seconds)','Control value')
        keys=('musical_energy','note_density','melody_pitch','percussion_intensity')
        for i,k in enumerate(keys):
            y=np.asarray(musical_features[k],float)
            if k=='melody_pitch':
                lo,hi=(float(np.nanmin(y)),float(np.nanmax(y))) if np.isfinite(y).any() else (0.,1.)
                y=(y-lo)/max(hi-lo,1.)
            elif np.isfinite(y).any() and np.nanmax(y)>0: y=y/np.nanmax(y)
            y=np.clip(y,0,1)
            _line(draw,music_box,mt,y,COLORS[i],(0,1)); draw.text((110+i*240,930),k.replace('_',' '),fill=COLORS[i])
    image.save(out/'earth_jukebox_features.png')
    if midi:
        roll=Image.new('RGB',(1200,720),'white'); d=ImageDraw.Draw(roll)
        x0,y0,x1,y1=100,45,1160,480; d.rectangle((x0,y0,x1,y1),outline=(120,120,120))
        allnotes=[n for group in midi for n in group]
        melodic_notes=[n for group in midi[:4] for n in group]
        plot_duration=max((n[1] for n in allnotes),default=max(float(t[-1]-t[0]) if len(t)>1 else 1.,1.))
        if melodic_notes:
            max_t=plot_duration; low=min(n[2] for n in melodic_notes); high=max(n[2] for n in melodic_notes)
            span=max(1,high-low)
            for j,group in enumerate(midi[:4]):
                for st,en,pitch,_ in group:
                    xa=x0+st/max(max_t,1e-9)*(x1-x0); xb=x0+en/max(max_t,1e-9)*(x1-x0)
                    yy=y1-(pitch-low)/span*(y1-y0)
                    d.rectangle((int(xa),int(yy-3),max(int(xb),int(xa)+2),int(yy+3)),fill=COLORS[j])
            d.text((x0,y1+13),'Melody: blue   Pad: red   Pluck: green   Bass: purple',fill=(20,20,20))
            d.text((x0,8),f'Piano roll | {max_t:.2f} musical seconds | melodic MIDI pitch {low}–{high}',fill=(20,20,20))
        dy0,dy1=550,680; d.rectangle((x0,dy0,x1,dy1),outline=(120,120,120)); d.text((x0,dy0-20),'Drum lane (separate GM drum pitches)',fill=(20,20,20))
        drums=midi[4] if len(midi)>4 else []; drum_max=plot_duration
        drum_rows={36:dy0+20,38:dy0+60,42:dy0+100}
        for pitch,y in drum_rows.items():
            d.text((x0-30,y-6),str(pitch),fill=(20,20,20)); d.line((x0,y,x1,y),fill=(230,230,230))
        for st,en,pitch,_ in drums:
            y=drum_rows.get(pitch,dy0+20); xa=x0+st/max(drum_max,1e-9)*(x1-x0); xb=x0+en/max(drum_max,1e-9)*(x1-x0)
            d.rectangle((int(xa),y-5,max(int(xb),int(xa)+2),y+5),fill=(95,95,95))
        roll.save(out/'earth_jukebox_piano_roll.png')
