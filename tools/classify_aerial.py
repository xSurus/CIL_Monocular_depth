import argparse
import csv
import os
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

BATCH_SIZE = 1024


def encode_text(model, processor, prompts, device):
    inputs = processor(text=prompts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = model.text_projection(model.text_model(**inputs).pooler_output)
    feats = feats.mean(0, keepdim=True)
    return feats / feats.norm(dim=-1, keepdim=True)


def classify(data_dir, out_csv, pos_label, pos_prompts, neg_prompts,
             model_id='openai/clip-vit-large-patch14', neg_label='ground'):
    """Classify images and write results to out_csv. Safe to call directly (no subprocess)."""
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    pos_feats = encode_text(model, processor, pos_prompts, device)
    neg_feats = encode_text(model, processor, neg_prompts, device)

    data_dir_p = Path(data_dir)
    rgb_files  = sorted(str(p.relative_to(data_dir_p)) for p in data_dir_p.rglob('*_rgb.png'))
    print(f'Found {len(rgb_files)} images in {data_dir}')
    if not rgb_files:
        raise FileNotFoundError(f'No *_rgb.png files found under {data_dir}')

    pos_score_key = f'{pos_label}_score'
    neg_score_key = f'{neg_label}_score'
    results = []

    for i in range(0, len(rgb_files), BATCH_SIZE):
        batch_files = rgb_files[i:i + BATCH_SIZE]
        images      = [Image.open(os.path.join(data_dir, f)).convert('RGB') for f in batch_files]
        inputs      = processor(images=images, return_tensors='pt').to(device)

        with torch.no_grad():
            img_feats = model.get_image_features(**inputs)
            if not isinstance(img_feats, torch.Tensor):
                img_feats = model.visual_projection(model.vision_model(**inputs).pooler_output)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        pos_scores = (img_feats @ pos_feats.T).squeeze(1).cpu().float().numpy()
        neg_scores = (img_feats @ neg_feats.T).squeeze(1).cpu().float().numpy()

        for fname, p, n in zip(batch_files, pos_scores, neg_scores):
            results.append({
                'file': fname,
                pos_score_key: float(p),
                neg_score_key: float(n),
                'label': pos_label if p > n else neg_label,
            })

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', pos_score_key, neg_score_key, 'label'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved labels to {out_csv}')

    n_pos = sum(1 for r in results if r['label'] == pos_label)
    print(f'{pos_label}: {n_pos}/{len(results)} ({100*n_pos/len(results):.1f}%)')

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--out',         default='labels.csv')
    parser.add_argument('--model',       default='openai/clip-vit-large-patch14')
    parser.add_argument('--pos_label',   default='aerial')
    parser.add_argument('--neg_label',   default='ground')
    parser.add_argument('--pos_prompts', nargs='+', required=True)
    parser.add_argument('--neg_prompts', nargs='+', required=True)
    args = parser.parse_args()
    classify(args.data_dir, args.out, args.pos_label, args.pos_prompts, args.neg_prompts,
             model_id=args.model, neg_label=args.neg_label)


if __name__ == '__main__':
    main()
