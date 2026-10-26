# Calibrated early warning for near-miss anticipation (evidential + conformal)

Frozen Video-Swin features -> causal temporal transformer -> two evidential heads
(Dirichlet risk, Normal-Inverse-Gamma time-to-accident) -> conformal warning policy with a
false-alarm guarantee, benchmarked under cross-dataset and corruption shift.

## Layout
| file | purpose |
|---|---|
| `make_meta.py` | build `meta.csv` (paths, labels, accident frame, splits, domain) from your dataset folders |
| `extract_features.py` | frozen Video-Swin-T features, run once per dataset / corruption (`corruptions.py` for rain, fog, ...) |
| `data.py` | dataset over cached features |
| `models.py` | causal temporal transformer + heads |
| `uncertainty.py` | risk + uncertainty for softmax / MC-dropout / ensemble / evidential (+ TTA mean/variance) |
| `losses.py` | targets, evidential (Sensoy) loss, NIG (Amini) loss, Gaussian NLL |
| `train.py` | train one model/seed; optional DANN (`dann.py`) |
| `evaluate.py` | AP@Δ, FAR/mTTA at fixed recall, ECE, Brier, risk-coverage, TTA MAE/coverage, conformal policy; optional TENT (`tta.py`) |
| `conformal.py` / `aci_demo.py` | split-conformal threshold + adaptive conformal under shift |
| `benchmark_latency.py` | single-pass vs MC-dropout vs ensemble cost |
| `tools/make_synthetic_features.py` | fake features to sanity-check the whole pipeline without data/GPU |

## 0. Sanity check (no data needed)
```bash
python tools/make_synthetic_features.py --out feats/syn_src --seed 0
python tools/make_synthetic_features.py --out feats/syn_shift --shift 1.0 --seed 1
python train.py --feat_dir feats/syn_src --kind evidential --tta nig --epochs 15 --out runs/evi/s0.pt
python train.py --feat_dir feats/syn_src --kind softmax --tta gauss --epochs 15 --out runs/softmax/s0.pt
python evaluate.py --method softmax=softmax:runs/softmax/s0.pt --method evi=evidential:runs/evi/s0.pt \
  --calib feats/syn_src@calib --eval src=feats/syn_src@test --eval shift=feats/syn_shift@test --out syn.csv
```

## 1. Real data
```bash
python make_meta.py --root data/DAD --toa_csv dad_toa.csv --domain dad --out meta_dad.csv
python extract_features.py --meta meta_dad.csv --out feats/dad_clean
for c in fog rain motion_blur low_light gaussian_noise; do
  python extract_features.py --meta meta_dad.csv --out feats/dad_${c}3 --corruption $c --severity 3; done
# repeat for CCD (target domain): meta_ccd.csv -> feats/ccd_clean
```
Train on `feats/dad_clean` **clean only**; corrupted/CCD sets are for testing.

## 2. Train (5 seeds each, mean ± std in the paper)
```bash
for s in 0 1 2 3 4; do
  python train.py --feat_dir feats/dad_clean --kind softmax    --tta gauss --seed $s --out runs/softmax/s$s.pt
  python train.py --feat_dir feats/dad_clean --kind evidential --tta nig   --seed $s --out runs/evi/s$s.pt
done
```
Softmax seeds double as the deep ensemble (5 models) and the MC-dropout model (seed 0).

## 3. Evaluate
```bash
python evaluate.py \
  --method softmax=softmax:runs/softmax/s0.pt \
  --method mcdrop=mcdropout:runs/softmax/s0.pt \
  --method ens=ensemble:runs/softmax/s0.pt,runs/softmax/s1.pt,runs/softmax/s2.pt,runs/softmax/s3.pt,runs/softmax/s4.pt \
  --method evi=evidential:runs/evi/s0.pt \
  --calib feats/dad_clean@calib \
  --eval dad=feats/dad_clean@test --eval ccd=feats/ccd_clean@test --eval dad_fog3=feats/dad_fog3@test \
  --alpha 0.1 --recall 0.8 --out results_seed0.csv
python aci_demo.py --method evi=evidential:runs/evi/s0.pt --calib feats/dad_clean@calib --target feats/ccd_clean@test
python benchmark_latency.py
```
Repeat evaluation per seed (swap the checkpoint paths) and aggregate the CSVs.

## Things to know before trusting a number
* The accident frame (`toa_frame`) must come from the dataset annotation. `make_meta.py`'s fallback is only a rough guess.
* Risk is "probability of an accident within `--horizon` seconds" (default 2.5 s), so ECE / Brier / conformal calibration are meaningful. Use `--label_mode weighted` for the DSA-style loss when comparing with published AP numbers.
* The conformal guarantee (false-alarm rate <= alpha on negative videos) assumes calibration and test negatives are exchangeable. Under shift it can be violated: that is a result worth reporting, and `aci_demo.py` shows the adaptive fix (needs false-alarm feedback).
* `--tent` and `--dann_dir` are experimental ablations, not the core claim.
* Tune the evidential KL weight (`--kl_coef`, `--kl_anneal`) and NIG regulariser (`--lam_reg`) on the validation split; evidential regression can be unstable.
* Class imbalance, per-dataset step rates and clip lengths differ; check `index.csv` after extraction.
