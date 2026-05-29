# Mitigating Distribution Shifts in Monocular Depth Estimation via Synthetic Data and Segment-Guided Refinement 

This repository contains our end-to-end monocular depth estimation pipeline for a benchmark with noticeable train-test distribution shift. Our base model is a frozen DINOv2 ViT-L/14 encoder with a trainable DPT Head that predicts both depth and confidence, inspired by the SOTA VGGT model. Rather than focusing only on architecture changes, our approach is driven by an exploratory dataset analysis: a CLIP-based scene audit reveals a pronounced mismatch between the training and test distributions, with vegetation emerging as the most weakly supported regime.
To address this, the pipeline combines several components. We first adapt the training data to the benchmark through aerial-image filtering, water-aware rebalancing, stratified validation, and standard augmentation. We then extend the training set with diffusion-generated overgrown images that preserve scene geometry while adding vegetation-heavy appearance. Finally, we add an optional SAM-guided refinement stage that is applied selectively to vegetation images through a CLIP-based gate. The overall strategy is therefore: diagnose the shifted regime, improve support for it during training, and refine predictions only where this targeted correction is beneficial.


---

## Quick Start (Colab)

> Runtime → Change runtime type → **A100 or G4**

### 1. Fill in Cell 1

Open `colab_train.ipynb` and set the five values in **Cell 1**:

| Variable | What to set |
|---|---|
| `REPO_URL` | `'git@github.com:xSurus/CIL_Monocular_depth.git'` |
| `BRANCH` | `'master'` |
| `CONFIG` | config path — see table below |
| `KAGGLE_TOKEN` | your Kaggle API token JSON string |
| `SUBMIT` | `False` (flip to `True` only to actually submit) |

Optional overrides in the same cell:

| Variable | Default | When to change |
|---|---|---|
| `BASE_CHECKPOINT` | `None` | point to an existing `.pt` to skip training |
| `FORCE_RETRAIN_REFINER` | `True` | set `False` to reuse an existing refiner checkpoint |
| `FORCE_REBUILD_VEGLIST` | `False` | set `True` to re-run CLIP vegetation scoring |

### 2. Pick a config

| Config | What it runs |
|---|---|
| `configs/overgrown_conf_12ep_bs40_data_aug_refiner.yaml` | **Best** — synthetic data + SAM refinement |
| `configs/overgrown_conf_12ep_bs40_data_aug.yaml` | Base model + synthetic data, no refinement |
| `configs/conf_12ep_bs40_data_aug.yaml` | Base model, no synthetic data |
| `configs/baseline_raw_train_12ep_bs40.yaml` | Raw baseline — no filtering, no augmentation |

### 3. Run all cells top to bottom

The notebook will:
1. Mount Google Drive
2. Load or generate an SSH key for GitHub access
3. Clone the repo into `/content/CIL_Monocular_depth`
4. Download the Kaggle dataset to local Colab SSD
5. Extract the overgrown training zip from Drive
6. Patch config paths to point at Colab-local data
7. Compute or restore CLIP scene-category CSVs
8. Train the base model → saves best weights to `checkpoints/{name}_best.pt`
9. *(if config has `refiner`)* Precompute depth cache → train SAM refiner
10. Run test predictions, optional VGGT vegetation swap, build submission CSV
11. Submit to Kaggle if `SUBMIT=True`

---

## Config

All training behaviour is set and fixed inside the YAML config files. The paths (`data_dir`, `extra_train_dir`, `filter_aerial_csv`, `water_csv`) are **patched automatically** by the notebook and are never edited by hand.

Examples of relevant parameters:

```yaml
epochs:     12        # training epochs
batch_size: 40        # reduce if OOM
lr:         0.0003    # learning rate

augmentation:
  hflip:        0.5                    # horizontal flip probability
  color_jitter: [0.2, 0.2, 0.2, 0.05] # brightness/contrast/saturation/hue

head:
  use_conf: true       # aleatoric confidence head (keep true for best configs)

refiner:               # present only in *_refiner configs
  epochs:     5
  batch_size: 64
  lr:         0.0001
```

---

## Prerequisites on Drive

Before running, your Google Drive (`MyDrive/CIL/`) must contain:

- `data/overgrown_train.zip` — synthetic overgrown training images
- `data/sam_v2/train_sam_v2/` — precomputed SAM v2 `.npz` feature files for training set (only needed for refiner configs)
- `data/sam_v2/test_sam_v2/` — same for test set (only needed for refiner inference)

Kaggle data is downloaded automatically via the API token.

---

## AI Usage Declaration

**Tool used:** Gemini 2.5 Pro
**Files affected:** `exploratory_data_analysis.ipynb`
**Purpose:** Helped generate plots quickly and tracked down a few bugs in the EDA cells.

---

**Tool used:** Gemini 2.0 Flash
**Files affected:** `generate_overgrown_data.ipynb`
**Purpose:** Helped debug the ControlNet/StableDiffusion setup and tested different prompt phrasings for the overgrown scene generation.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `colab_train.ipynb`, `tools/colab_notebook_utils.py`
**Purpose:** We had Claude write the repetitive Colab cell setup , Drive mounting, SSH key handling, Kaggle download, path patching. The utility functions for archiving checkpoints to Drive and building the submission CSV. We designed the overall notebook flow ourselves.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/visualize_predictions.py`, `tools/plot_distribution.py`
**Purpose:** Purely plotting scripts. We had Claude write them so we could focus on the model rather than matplotlib syntax.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/create_submission.py`
**Purpose:** The Kaggle submission format (base64-encoded depth arrays, zlib compression, CSV layout) had a precise spec. We had Claude implement it from that spec.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/train.py`
**Purpose:** Claude wrote the training loop plumbing: mixed-precision (AMP) setup, gradient scaling, and checkpoint save/load. We focused on the loss functions and model design.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/repro.py`
**Purpose:** Asked Claude to write the seed setup across `random`, `numpy`, and `torch`, including the per-worker seed hook for DataLoaders — we knew what we needed but not the exact API.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/dataset.py`
**Purpose:** We wrote the dataset logic. Claude helped with the `torchvision.transforms` composition syntax.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/losses.py`
**Purpose:** We derived the loss functions ourselves. Claude added comments explaining the masking and clamping choices, and double-checked the confidence-weighting formula.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/model.py`
**Purpose:** Claude wrote the `DepthModel` wrapper and its docstring.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/pipeline/encoders.py`
**Purpose:** We knew we wanted to pull patch tokens from an intermediate DINOv2 block, but weren't sure of the `timm` API for custom image sizes and forward hooks. Claude figured that part out.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/pipeline/heads.py`
**Purpose:** We designed the `AggregatorDPTHead` architecture. Claude translated the sincos positional embedding math into working PyTorch, specifically the `einsum` notation and `linspace` indexing we kept getting wrong.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/pipeline/refiner.py`
**Purpose:** The blending design was ours. Claude wrote the module structure and the docstring.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/precompute_depth.py`, `src/predict.py`, `src/predict_refiner.py`
**Purpose:** All three follow the same inference pattern. Claude wrote them (arg parsing, DataLoader setup, autocast context, checkpoint loading with `_orig_mod.` prefix stripping, `.npy` saving).

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `src/train_refiner.py`
**Purpose:** Same situation as `src/train.py`, Claude wrote the training loop; we wrote the loss and data loading logic specific to the SAM refiner.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/classify_aerial.py`, `tools/clip_scene_scoring.py`, `tools/build_vegetation_list.py`
**Purpose:** We knew what we wanted from CLIP (batch image/text encoding, cosine similarity against prompt sets) but hadn't used the HuggingFace API before. Claude wrote these scripts from our description of the scoring logic.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/merge_categories.py`
**Purpose:** Claude wrote this. It's a straightforward pandas merge of the aerial, water, and vegetation label CSVs.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/swap_predictions.py`
**Purpose:** Claude wrote the script that swaps base-model depth outputs with VGGT predictions for flagged vegetation images.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/predict_vggt.py`
**Purpose:** We wrote the inference logic. Claude translated the VGGT repo's example code into a standalone script and handled the depth tensor resizing.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `tools/colab_sam_preprocessing.py`
**Purpose:** Claude wrote the concurrent download and SAM preprocessing helper — the `ThreadPoolExecutor` batching and subprocess calls to the SAM2 scripts.

---

**Tool used:** Claude Code (Claude Sonnet 4.6)
**Files affected:** `sam_segmentation_pipeline.ipynb`
**Purpose:** Claude wrote the Colab notebook cells for installing SAM2 and running the preprocessing pipeline from `tools/colab_sam_preprocessing.py`.

---

In-line completion was used throughout the project for formatting, debugging documentation.
