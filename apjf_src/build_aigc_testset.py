"""
AIGC 测试集生成器
用 Stable Diffusion Inpainting 在 COCO val 上生成 AI 篡改图

用法:
  /home/oyp/miniconda3/envs/apjf/bin/python -u apjf_src/build_aigc_testset.py
"""
import sys, os, gc, random
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms as T
from apjf_src.joint_config import VAL_DIR

OUT_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/aigc_test'
AUTH_DIR = os.path.join(OUT_DIR, 'authentic')
FORG_DIR = os.path.join(OUT_DIR, 'forgery')
MASK_DIR = os.path.join(OUT_DIR, 'masks')
NUM_SAMPLES = 200
IMG_SIZE = 512
SEED = 42

def generate_mask(w, h, img_size=512):
    """生成随机不规则掩膜"""
    mask = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask)
    x = random.randint(0, img_size // 3)
    y = random.randint(0, img_size // 3)
    rw = random.randint(img_size // 4, img_size // 2)
    rh = random.randint(img_size // 4, img_size // 2)
    draw.rectangle([x, y, x+rw, y+rh], fill=255)
    return mask

def main():
    os.makedirs(AUTH_DIR, exist_ok=True)
    os.makedirs(FORG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    # 收集 COCO val 图片
    all_imgs = [os.path.join(VAL_DIR, f) for f in sorted(os.listdir(VAL_DIR))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(SEED)
    random.shuffle(all_imgs)
    selected = all_imgs[:NUM_SAMPLES]
    print(f'[AIGC] 已选 {len(selected)} 张 COCO val 图')

    # 加载 SD Inpaint
    print('[AIGC] 加载 Stable Diffusion Inpainting...')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print('[AIGC] 模型加载完成')

    prompt = 'a realistic photo, natural lighting, consistent style'
    negative = 'blurry, unnatural, artifact, painting, cartoon'

    for i, img_path in enumerate(tqdm(selected, desc='生成 AIGC 篡改图')):
        # 加载原图并缩放到 512
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        base_name = f'{i:04d}'

        # 保存原图（可信样本）
        img.save(os.path.join(AUTH_DIR, f'{base_name}.jpg'), quality=95)

        # 生成掩膜
        mask = generate_mask(IMG_SIZE, IMG_SIZE, IMG_SIZE)
        mask.save(os.path.join(MASK_DIR, f'{base_name}_mask.png'))

        # SD Inpaint 生成篡改区域
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=img,
            mask_image=mask,
            height=IMG_SIZE,
            width=IMG_SIZE,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

        # 保存篡改图
        result.save(os.path.join(FORG_DIR, f'{base_name}.jpg'), quality=95)

        # 定期清理
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    pipe.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    print(f'\n[AIGC] 完成! 共生成:')
    print(f'  Authentic: {len(os.listdir(AUTH_DIR))} 张 (原图)')
    print(f'  Forgery:   {len(os.listdir(FORG_DIR))} 张 (SD Inpaint 篡改)')
    print(f'  Masks:     {len(os.listdir(MASK_DIR))} 张')
    print(f'  保存路径: {OUT_DIR}')

if __name__ == '__main__':
    main()
