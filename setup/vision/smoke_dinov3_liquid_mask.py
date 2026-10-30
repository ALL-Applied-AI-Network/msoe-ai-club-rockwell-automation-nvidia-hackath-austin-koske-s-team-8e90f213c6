"""CPU-only native-backbone shape/reload check; no fitting or real images."""
import argparse
import os
from pathlib import Path

from train_pair_masks import sha,write


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backbone',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    from dinov3_liquid_mask import DinoLiquidMask
    torch.set_num_threads(4);torch.manual_seed(19);current=torch.rand(1,3,192,112);reference=torch.rand_like(current);anchor=torch.full((1,3),.4);checks={}
    for mode in ('rgb','paired'):
        model=DinoLiquidMask(a.backbone,mode).cpu().eval()
        with torch.inference_mode():mask,presence=model(current,reference,anchor,anchor)
        assert mask.shape==(1,192,112) and presence.shape==(1,) and torch.isfinite(mask).all() and torch.isfinite(presence).all()
        assert not any(p.requires_grad for p in model.encoder.parameters())
        model.train();assert not model.encoder.training;model.eval();checkpoint=a.out/(mode+'-decoder.pt');torch.save(model.decoder_state(),checkpoint)
        with torch.no_grad():
            for parameter in model.mask_head.parameters():parameter.add_(.1)
        model.load_decoder(torch.load(checkpoint,map_location='cpu',weights_only=True))
        with torch.inference_mode():reloaded=model(current,reference,anchor,anchor)
        assert torch.equal(mask,reloaded[0]) and torch.equal(presence,reloaded[1])
        checks[mode]=dict(mask_shape=list(mask.shape),presence_shape=list(presence.shape),exact_decoder_reload=True,
            encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write(a.out/'completed.json',dict(cpu_only=True,fitting=False,real_images=False,checks=checks,torch=torch.__version__,
        helper_sha256=sha(__file__),model_helper_sha256=sha(Path(__file__).with_name('dinov3_liquid_mask.py'))))
    print(checks)
