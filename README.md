# Facial-Expression Classification under Image Degradation

STOR 566 (Introduction to Deep Learning, Fall 2026, UNC-Chapel Hill) - Group 9  
Dominic Jiang, Alyson Ficca, Emily Nguyen, Isaiah Lu

Facial-expression classifiers are usually compared on clean test images, whereas practical inputs from cheap cameras or distant crops are often noisy or low-resolution. On FER-2013, we compare three models (a 94K-parameter CNN trained from scratch, ImageNet-pretrained ResNet-18, and ImageNet-pretrained ViT-B/16) across five levels of Gaussian noise and five levels of reduced resolution, and test whether Gaussian-noise augmentation improves robustness to both.

## Results

Test accuracy (%), mean ± standard deviation over three seeds, on the deduplicated test split (3,268 images, majority-class rate 25.9%). "Noise avg." is the mean over sigma in {0.025, 0.05, 0.10, 0.15, 0.20}; "Resolution avg." is the mean over 40, 32, 24, 16, and 8 px.

| Model | Training | Original | Noise avg. | Resolution avg. |
|---|---|---|---|---|
| Baseline CNN (94K parameters, from scratch) | standard | 53.2 ± 1.3 | 37.5 ± 0.4 | 38.0 ± 0.4 |
|  | noise augmentation | 53.5 ± 0.4 | 47.5 ± 0.5 | 39.4 ± 0.3 |
|  | standard, conventional checkpoint | 53.9 ± 0.5 | 34.4 ± 3.6 | 38.7 ± 2.1 |
| ResNet-18 (ImageNet-pretrained) | standard | 61.2 ± 1.4 | 37.8 ± 2.5 | 38.6 ± 0.9 |
|  | noise augmentation | 63.1 ± 1.1 | 56.7 ± 0.4 | 47.7 ± 2.3 |
|  | standard, conventional checkpoint | 63.7 ± 0.3 | 33.7 ± 2.8 | 39.7 ± 1.2 |
| ViT-B/16 (ImageNet-pretrained) | standard | 69.3 ± 1.1 | 44.0 ± 0.3 | 45.8 ± 0.9 |
|  | noise augmentation | 68.5 ± 0.5 | 61.9 ± 0.1 | 49.8 ± 0.7 |
|  | standard, conventional checkpoint | 69.5 ± 0.4 | 43.2 ± 0.3 | 45.7 ± 0.4 |

"Standard" and "noise augmentation" follow the validation selection rule (`S = 0.5 * Acc_val(original) + 0.5 * Acc_val(noisy)`). "Conventional checkpoint" repeats standard training with the checkpoint chosen on original validation accuracy only. Reduced-resolution images are not used for model selection.

## Repository Structure

| Path | Content |
|---|---|
| `src/ferlib.py` | Shared pipeline: duplicate filtering, model definitions, degradations, training, selection rule, evaluation |
| `src/prep_data.py` | Builds deduplicated FER-2013 train/val/test splits from `fer2013.csv` or `icml_face_data.csv` |
| `src/analyze.py` | Aggregates per-run JSON metrics into summary tables and degradation/learning-curve plots |
| `src/analyze_extra.py` | Computes prediction collapse shares, class distributions, recall retention, and bootstrap summaries |
| `notebooks/STOR566_Group9_FER_experiments.ipynb` | Main Colab/Kaggle GPU notebook (tuning, 18 main runs, CNN and ResNet-18 reference runs) |
| `notebooks/STOR566_Group9_FER_remaining_runs.ipynb` | Companion Kaggle/Colab GPU notebook (ViT-B/16 reference runs and post-hoc learning-rate check) |
| `results/` | Per-run JSON records (`final/`, `tune/`, `posthoc/`), saved hyperparameter selections (`selection_*.json`), `data_audit.json`, and summary tables (`analysis/`, `analysis_extra/`) |
| `requirements.txt` | Python package dependencies |
| [`STOR566_Group9_Final_Report.pdf`](STOR566_Group9_Final_Report.pdf) | Final project report (NeurIPS format) |
| [`STOR566_Group9_Final_Presentation.pdf`](STOR566_Group9_Final_Presentation.pdf) | Final presentation slides |

## Reproducing

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset. FER-2013 is not redistributed here. To build the deduplicated splits locally from `fer2013.csv` or `icml_face_data.csv` (27,418 train / 3,291 val / 3,268 test images; content fingerprint recorded as `data_fingerprint` in each run JSON):

   ```bash
   python3 src/prep_data.py path/to/fer2013.csv data/fer_dedup.npz
   ```

3. Run GPU experiments. Open `notebooks/STOR566_Group9_FER_experiments.ipynb` and `notebooks/STOR566_Group9_FER_remaining_runs.ipynb` in Google Colab or Kaggle with a T4 GPU and run all cells. Both notebooks are committed with cleared outputs; all finished run metrics are already saved under `results/`.

4. Summarize results (CPU only):

   ```bash
   python3 src/analyze.py results
   python3 src/analyze_extra.py results data/fer_dedup.npz
   ```

   `analyze.py` regenerates `results/analysis/` from the committed run JSON files. `analyze_extra.py` also requires `data/fer_dedup.npz` and the `results/final/*_preds.npz` files produced during GPU runs (not committed due to size) to recompute the Section 3 bootstrap intervals.
