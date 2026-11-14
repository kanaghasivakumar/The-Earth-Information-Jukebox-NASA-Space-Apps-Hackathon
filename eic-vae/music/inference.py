from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def extract_latents(checkpoint_path, data_root, output_dir, *, batch_size=8, device='cpu', cache_dir=None):
    """Load an existing checkpoint and deterministically extract posterior means."""
    from dataset import NASAVisualizationDataset
    from models.vae import ConvVAE
    try: ckpt=torch.load(checkpoint_path,map_location=device,weights_only=True)
    except TypeError: ckpt=torch.load(checkpoint_path,map_location=device)
    prep=dict(ckpt['preprocessing']); cfg=ckpt['model_config']; size=int(cfg.get('image_size',cfg.get('size',64)))
    # Keep checkpoint preprocessing authoritative; cache location is operational only.
    ds=NASAVisualizationDataset(data_root,**prep,cache_dir=cache_dir)
    if len(ds.sources)!=1: raise ValueError(f'Expected one animation source for one composition; found {len(ds.sources)}. Pass a single video or source folder.')
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model=ConvVAE(latent_dim=int(cfg['latent_dim']),image_size=size).to(device)
    state=ckpt.get('model',ckpt.get('model_state_dict',ckpt.get('model_state')))
    if state is None: raise KeyError('checkpoint has no model weights')
    model.load_state_dict(state); model.eval()
    means=[]; logvars=[]; times=[]; flow_magnitudes=[]
    with torch.no_grad():
        for batch in loader:
            x=batch['input'].to(device)
            mu,lv=model.encode(x); means.append(mu.cpu().numpy()); logvars.append(lv.cpu().numpy()); times.append(batch['timestamp'].numpy())
            flow_magnitudes.append(torch.sqrt(torch.sum(x[:,3:5]**2,dim=1)).mean(dim=(1,2)).cpu().numpy())
    means=np.concatenate(means); logvars=np.concatenate(logvars); times=np.concatenate(times)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    np.save(out/'latent_means.npy',means); np.save(out/'latent_logvars.npy',logvars); np.save(out/'timestamps.npy',times)
    return {'latent_means':means,'latent_logvars':logvars,'timestamps':times,'flow_magnitude':np.concatenate(flow_magnitudes),'preprocessing':prep,'dataset':ds}
