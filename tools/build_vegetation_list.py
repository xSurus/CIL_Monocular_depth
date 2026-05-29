import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

TEST_DIR   = '/content/data/monodepth_kaggle2026/test'
OUT        = 'src/vegetation_list.txt'

CATEGORIES = ["Aerial", "Architecture", "Vegetation"]
TEXT_PROMPTS = [
    "An aerial view looking down from far away",
    "A photo dominated by buildings and architecture",
    "A photo heavily featuring trees, bushes, and vegetation",
]

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

fnames = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith('.png'))

@torch.no_grad()
def classify(fnames, model, processor):
    vegetation = []
    for fname in tqdm(fnames, desc="CLIP"):
        image  = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
        inputs = processor(text=TEXT_PROMPTS, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        probs  = model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()[0]
        if CATEGORIES[np.argmax(probs)] == "Vegetation":
            vegetation.append(fname)
    return vegetation

vegetation = classify(fnames, model, processor)

del model, processor
torch.cuda.empty_cache()

print(f'{len(vegetation)}/{len(fnames)} vegetation ({100*len(vegetation)/len(fnames):.1f}%)')
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, 'w') as f:
    for rank, fname in enumerate(vegetation, 1):
        f.write(f"  {rank}. {fname}  [{os.path.join(TEST_DIR, fname)}]\n")
print(f'Written to {OUT}')
