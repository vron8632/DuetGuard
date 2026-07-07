"""
AAAI 论文定性对比图生成 (修复文本截断)
用法: /home/oyp/miniconda3/envs/apjf/bin/python -u apjf_src/fig_qualitative.py
"""
import sys, os
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch; import torch.nn.functional as F; import numpy as np
from PIL import Image, ImageDraw, ImageFont; from torchvision import transforms as T; from torch.utils.data import DataLoader
from apjf_src.models.adapter import WatermarkBranch
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.models.fusion_layer import FusionLayer
from apjf_src.joint_config import *
from apjf_src.run_cvpr import EvalDataset, eval_batch

DEVICE='cuda'
OUT='/media/oyp/数据/Projects/042_image_forensic/DuetGuard/aaai-duetguard/fig'
os.makedirs(OUT,exist_ok=True)
SIZE=160; gap=3; ROW_H=24  # compact layout

def tensor_pil(t):
    n=(t.cpu().permute(1,2,0).numpy()*255).astype(np.uint8).clip(0,255); return Image.fromarray(n)
def heatmap_pil(t2d):
    t=t2d.cpu(); t=(t-t.min())/(t.max()-t.min()+1e-8)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    c=plt.get_cmap('jet')((t.numpy()*255).astype(np.uint8))[...,:3]; return Image.fromarray((c*255).astype(np.uint8))

WM=WatermarkBranch(OMNIGUARD_CKPT,DEVICE).eval()
for p in WM.parameters(): p.requires_grad=False
SP=SPNExtractor(fp_dim=128,base_ch=64,num_blocks=2,expand_ratio=4).to(DEVICE)
SP.load_state_dict(torch.load(os.path.join(SAVE_DIR,'spn_finetuned_best.pth'),map_location='cpu',weights_only=True)); SP.eval()
FU=FusionLayer(fp_dim=128,act_feat_dim=192,num_classes=3).to(DEVICE)
FU.load_state_dict(torch.load(os.path.join(SAVE_DIR,'fusion_best.pth'),map_location='cpu',weights_only=True)); FU.eval()

@torch.no_grad()
def predict(wm,spn,fusion,img_tensor):
    sec=torch.ones_like(img_tensor)*0.5
    stego,snoisy,rec=wm.forward(img_tensor,sec,True); sf,nm=spn(snoisy)
    d=(rec-sec).abs(); af=F.adaptive_avg_pool2d(d,(8,8)).reshape(img_tensor.size(0),-1)
    q=(1.0-d.mean(dim=[1,2,3])).clamp(0,1); p=nm.std(dim=[1,2,3],keepdim=True)
    logits,_=fusion({'active_feat':af,'quality':q},{'spn_feat':sf,'pce':p})
    return logits.argmax(dim=1).item(), d, nm

def get_batch(seed,n=8):
    ds=EvalDataset(VAL_DIR,num_samples=n,seed=seed)
    return next(iter(DataLoader(ds,batch_size=n,shuffle=False)))

vn={0:'Auth',2:'Forg'}
try: fnt=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except: fnt=None

for seed_val,fig_rows,fig_name,fig_headers,ncols in [
    (100,[2,4,6,1],'ablation_qual.png',['Original','P3 Full','P0(No SPN)'],3),
    (200,[1,3,5,7],'visualization.png',['Input','Active','Passive','Prediction'],4),
]:
    batch=get_batch(seed_val,8)
    _,_,_,_,final=eval_batch(WM,SP,FU,batch)
    diffs=[]; noises=[]; preds=[]
    for i in range(8):
        p,dd,nn=predict(WM,SP,FU,final[i:i+1]); preds.append(p); diffs.append(dd[0]); noises.append(nn[0])

    n_r=len(fig_rows)
    # Extra width for text annotations on rightmost columns
    extra_w=50 if fig_name=='visualization.png' else 40
    w=ncols*(SIZE+gap)+gap+extra_w
    h=n_r*(SIZE+gap)+gap+ROW_H+10
    fig=Image.new('RGB',(w,h),(255,255,255)); draw=ImageDraw.Draw(fig)
    for ci,hdr in enumerate(fig_headers):
        draw.text((ci*(SIZE+gap)+gap+5,2),hdr,fill=(30,30,30),font=fnt)
    for ri,idx in enumerate(fig_rows):
        y=(ri+1)*gap+ri*SIZE+ROW_H
        # Column 1: input image
        fig.paste(tensor_pil(final[idx]).resize((SIZE,SIZE)),(gap,y))
        # Column 2: heatmap
        fig.paste(heatmap_pil(diffs[idx].mean(dim=0)).resize((SIZE,SIZE)),(SIZE+2*gap,y))

        pred='Auth' if preds[idx]==0 else 'Forg'; gt=vn.get(batch['verdict'][idx].item(),'?')
        c_green=(0,130,0); c_red=(200,0,0)

        if fig_name=='ablation_qual.png':
            # Compute P0 predictions for this row
            fi=final[idx:idx+1]; sec_i=torch.ones_like(fi)*0.5
            _,si,ri=WM.forward(fi,sec_i,True); sf,_=SP(si)
            d0=(ri-sec_i).abs(); af0=F.adaptive_avg_pool2d(d0,(8,8)).reshape(1,-1)
            logits_p0,_=FU({'active_feat':af0,'quality':(1.0-d0.mean()).clamp(0,1)},
                           {'spn_feat':sf,'pce':torch.tensor(0.0,device=DEVICE)})
            p0='Auth' if logits_p0.argmax(dim=1).item()==0 else 'Forg'
            # Col 3: P0 comparison image
            _,_,_,_,fp0=eval_batch(WM,SP,FU,batch)
            fig.paste(tensor_pil(fp0[idx]).resize((SIZE,SIZE)),(2*SIZE+3*gap+extra_w//2,y))
            # Labels: GT on col0, P3 on col1, P0 on col2
            draw.text((gap+4,y+SIZE-24),f'GT:{gt}',
                      fill=(0,0,0),font=fnt,stroke_width=2,stroke_fill=(255,255,255))
            draw.text((SIZE+2*gap+4,y+SIZE-24),f'P3:{pred}',
                      fill=c_green if pred==gt else c_red,font=fnt,stroke_width=2,stroke_fill=(255,255,255))
            draw.text((2*SIZE+3*gap+extra_w//2+4,y+SIZE-24),f'P0:{p0}',
                      fill=c_green if p0==gt else c_red,font=fnt,stroke_width=2,stroke_fill=(255,255,255))

        elif fig_name=='visualization.png':
            # Col 3: passive heatmap, Col 4: text prediction
            fig.paste(heatmap_pil(noises[idx].mean(dim=0)).resize((SIZE,SIZE)),(2*SIZE+3*gap+10,y))
            # Overlay text on col 4 cell
            draw.text((3*SIZE+4*gap+15,y+SIZE-48),f'GT:{gt}',fill=(0,0,0),
                      font=fnt,stroke_width=2,stroke_fill=(255,255,255))
            draw.text((3*SIZE+4*gap+15,y+SIZE-24),f'Pred:{pred}',
                      fill=c_green if pred==gt else c_red,font=fnt,
                      stroke_width=2,stroke_fill=(255,255,255))

    fig.save(os.path.join(OUT,fig_name))
    print(f'{fig_name}: {w}x{h} ({os.path.getsize(os.path.join(OUT,fig_name))//1024}KB)')
print('Done!')
