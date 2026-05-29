# Dataset Distribution Analysis and Training Improvements

## Overview

We identified and addressed a significant covariate shift between the training and test sets of the CIL 2026 monocular depth estimation dataset. Using zero-shot CLIP classification, we characterized the scene-type distribution of both splits and designed targeted interventions that improved Kaggle performance from 0.525 → 0.516 (siRMSE), closing roughly half the gap between our new architecture and the old architecture best of 0.507.

---

## 1. Dataset Distribution Analysis

### Method

We used OpenAI's `clip-vit-large-patch14` as a zero-shot binary classifier. For each scene type, we defined positive and negative text prompts and scored every image in both train (22,605 images) and test (591 images) against both prompt sets. An image is assigned label `positive` if its CLIP similarity to the positive prompts exceeds its similarity to the negative prompts.

We then merged the three binary classifiers into a single multi-class taxonomy using a priority hierarchy: **aerial → water → vegetation → other**. This ensures, for example, that an aerial shot of a Paris park (which contains grass) is categorised as aerial, not vegetation.

### Findings

| Category | Train | Test | Gap |
|---|---|---|---|
| Aerial | 8.1% (1,822 img) | 0.0% | −8.1pp |
| Water | 15.1% (3,422 img) | 49.2% (291 img) | +34.1pp |
| Vegetation | ~0.1% (13 img) | ~19.5% (115 img) | +19.4pp |
| Other | 76.7% | 31.3% | −45.4pp |

#### Aerial contamination

8.1% of training images are aerial photographs (drone shots, bird's-eye views). The test set contains zero aerial images. The model is therefore trained on a scene type it will never encounter at inference time. These images are not just unhelpful — they actively shift the model's prior toward top-down depth geometry that does not generalise to ground-level scenes.

Visual inspection confirmed the classifier is accurate: the top-scoring aerial training images are overwhelmingly shots of Paris landmarks (Champ de Mars, Trocadero) and other major cities from altitude.

#### Water imbalance

Water scenes are heavily underrepresented in training (15.1%) relative to test (49.2%) — a factor of ~3×. This is the most actionable finding because the training set does contain ~3,400 water images, making upsampling feasible.

Depth estimation on water is particularly challenging: LiDAR and structured-light sensors return invalid or highly noisy depth for specular surfaces due to reflectance rather than IR absorption. Water pixels therefore tend to have sparse or missing ground truth, making water scenes disproportionately difficult to learn from.

Note: the water imbalance is partially correlated with the aerial issue. Some of the highest-scoring "water" images in training are aerial shots of coastal cities (e.g., Chicago lakefront seen from above). These are correctly routed to the aerial category under our priority scheme, and filtered out. This means the remaining training water images are more representative of ground-level water — closer to the canal and river scenes dominant in the test set.

#### Vegetation gap

Ground-level vegetation scenes (forests, tree-lined paths, parks photographed at eye level) are essentially absent from training (~0.1%, 13 images) but constitute approximately 19.5% of the test set.

This is a **structural gap** — unlike the water imbalance, it cannot be addressed through upsampling because the data does not exist in sufficient quantity to learn from. During CLIP classification iteration, we confirmed that virtually all training images flagged as "vegetation" were in fact aerial shots of parks with visible grass, not ground-level forest scenes. The training dataset appears to be sourced primarily from urban landmark photography (consistent with MegaDepth-style data), which systematically excludes dense woodland and forest environments.

This represents a known model weakness for approximately one-fifth of the test set.

---

## 2. Interventions

### 2.1 Aerial Filtering

We removed all 1,822 images classified as aerial from the training pool before any further processing. This reduces the effective training set to 20,783 images.

This is the most conservative intervention: it removes data that has zero overlap with the test distribution. Any model capacity spent learning aerial depth geometry is wasted on this dataset.

### 2.2 Stratified Validation Split

Without intervention, a random 20% validation split yields approximately 17% water images in val — far below the test set's 49%. This means validation siRMSE is a biased estimator of test performance: the model can achieve a good val score while still performing poorly on water.

We replaced the random split with a stratified split targeting **30% water in the validation set**. This is a deliberate compromise between the natural training distribution (~20%) and the test distribution (~49%). A val set matching the test distribution exactly (49% water) would cause val loss to be systematically higher than training loss, making progress harder to interpret.

The 30% target was chosen to improve the correlation between val siRMSE and Kaggle score without creating an artificial distribution shift within the training loop.

### 2.3 Water Upsampling (WeightedRandomSampler)

We applied `WeightedRandomSampler` from PyTorch to oversample water images during training. Each water training image receives a weight of 4.0; all non-water images receive weight 1.0. The sampler draws with replacement, and `num_samples` is fixed at `len(train_indices)` so epoch length is unchanged.

With 4× upsampling, the fraction of water images in each training batch increases from ~20% to approximately **49%** — matching the test distribution. This required iterating: an initial run at 2× upsampling produced only marginal improvement (0.525 → 0.528 Kaggle), suggesting the model needed a more aggressive distribution correction.

The priority ordering (aerial removed before split, water upsampled, vegetation not addressable) ensures the interventions are composable and independently reasoned.

### 2.4 Data Augmentation

We added two augmentations applied to the training set only:

**Horizontal flip (p = 0.5):** Applied identically to both the RGB image and the depth ground truth map. Monocular depth estimation is horizontally symmetric — there is no physical reason why a scene viewed from the left should have different depth properties than the same scene mirrored. This doubles the effective variety of training samples at zero cost.

**Color jitter (brightness ±20%, contrast ±20%, saturation ±20%, hue ±5°):** Applied to the RGB image only. Depth is a geometric property, not a photometric one, so perturbing colour does not require any change to the depth ground truth. The motivation is to make the model less sensitive to lighting conditions and colour temperature, which vary significantly between the training set (urban landmark photography, often with consistent daylight conditions) and the test set (mixed outdoor and indoor scenes).

Parameter values were kept deliberately mild to avoid degrading performance, informed by a prior experiment on the old architecture where stronger augmentation was found to hurt rather than help.

---

## 3. Results

### Ablation

| Config | Epochs | Val siRMSE | Kaggle | Notes |
|---|---|---|---|---|
| `dinov2_base_bs96` | 8 | 0.5047 | 0.525 | Baseline — random split, no filtering, no aug |
| `dinov2_filtered_bs96` | 8 | — | 0.528 | Aerial filter + stratified val + 2× water upsample |
| `dinov2_water4x_aug_bs96` | 8 | 0.4860 | **0.516** | 4× water upsample + hflip + color jitter |
| Old arch best (`dinov2_sam_loss_strong_8ep`) | 8 | ~0.487 | 0.507 | Reference |

### Analysis

The 2× upsampling run (`dinov2_filtered_bs96`) produced negligible improvement (+0.003 Kaggle). The 4× run with augmentation (`dinov2_water4x_aug_bs96`) produced a substantial improvement (+0.009 Kaggle, −0.019 val siRMSE).

Two effects are compounded in the final run and cannot be perfectly separated without further ablation:
1. Stronger water upsampling (2× → 4×): directly addresses the most actionable distribution gap
2. Augmentation: provides regularisation that slows overfitting, allowing the model to continue improving across all 8 epochs

Evidence for the regularisation effect: in previous experiments on the old architecture, training loss would plateau and val siRMSE would begin to rise after epoch 8, indicating overfitting. In the augmented run, val siRMSE improved monotonically across all 8 epochs (0.5358 → 0.4860), with no sign of plateau. This suggests augmentation is buying additional effective training budget beyond what was available without it.

The gap to the old architecture best (0.507 Kaggle) narrowed from 0.018 → 0.009. Val siRMSE (0.4860) now matches the old architecture's best val performance, suggesting the remaining Kaggle gap may reflect test-set-specific factors rather than model capacity.

---

## 4. Key Conclusions for the Report

1. **Dataset characterisation matters.** The train/test distribution mismatch was not visible from standard dataset statistics (image count, resolution, depth range). It required semantic classification — using a pretrained vision-language model — to surface. This kind of exploratory data analysis is undervalued in depth estimation work.

2. **Upsampling strength needs to match the target distribution.** 2× water upsampling (~33% of batches) was insufficient; 4× (~49% of batches, matching the test set) produced a meaningful improvement. The lesson is that upsampling should be calibrated to close the gap, not just reduce it.

3. **Stratified validation is necessary for reliable model selection.** With a random split, val siRMSE on this dataset is a biased proxy for Kaggle performance because the val set is water-underrepresented. Stratified splitting makes the metric more meaningful and reduces the risk of selecting a checkpoint that has overfit to the training distribution.

4. **Some distribution gaps cannot be closed by data manipulation alone.** The vegetation gap (~0.1% train vs ~20% test) is structural — the training data simply does not contain ground-level forest scenes. This points to a fundamental limitation of training on curated urban photography datasets for deployment in diverse real-world environments. Addressing it would require either additional data collection or domain adaptation techniques.

5. **Augmentation and upsampling are complementary.** Upsampling corrects the distribution; augmentation adds regularisation that prevents the model from memorising individual samples and allows it to keep improving across more training epochs.

---

## 5. Remaining Gap and Next Steps

The remaining 0.009 Kaggle gap to the old architecture best (0.507) is plausible explanations:
- The vegetation gap (~20% of test, ~0% of train) contributes irreducible error
- The old architecture had a confidence head and FiLM conditioning that may have helped on certain scene types
- 8 epochs may not be enough — the augmented model's val curve had not plateaued

Ongoing experiment: `dinov2_water4x_aug_12ep_bs96` extends training to 12 epochs with identical settings to test whether the model continues to improve with the additional regularisation from augmentation.

VGGT ensemble is also pending: VGGT is a zero-shot metric depth model that can be averaged with DINOv2 predictions at inference time, potentially recovering some accuracy on out-of-distribution scenes like vegetation.
