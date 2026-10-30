# Measuring and Predicting Penetration and Dilution in Directed Energy Deposition

Code for the B.Tech project of Ayush Agarwal (Department of Mechanical Engineering, IIT Guwahati),
supervised by Prof. S. N. Joshi.

In directed energy deposition (DED), the bead height `H` and width `W` can be measured during printing,
but the penetration depth `D`, the remelted substrate area and the dilution need an etched cross-section.
This repository contains a pipeline that

1. **segments cross-section micrographs** into clad and remelt regions and measures `H`, `W`, `D`,
   clad area `A_d`, remelt area `A_s` and dilution `A_s / (A_d + A_s)`;
2. **checks every automatic measurement against expert measurements** and uses only the sections that agree
   as training labels for a **U-Net**;
3. **learns process-to-geometry models** (Gaussian process regression, random forest, log-linear regression
   on physically motivated inputs)
   and evaluates them on **unseen settings** and under **extrapolation in laser power**, not only on random
   splits;
4. tests **few-shot transfer** of both the segmentation network and the process model to a new domain;
5. tests whether a **coaxial melt-pool camera** adds information on penetration.

It is developed on the public 316L powder-DED dataset of Akhavan et al. and applied to laser-wire DED
micrographs of Inconel 718 on SS304.

## Main results (public powder-DED dataset)

| Result | Value |
|---|---|
| Rule-based segmentation, 393 expert-agreeing sections | MAE `H` 33 µm, `W` 39 µm, `D` 31 µm |
| Self-trained U-Net | within ~10 µm of the rule-based accuracy on unseen tracks; recovers 301 sections the rules missed |
| Sections with validated area and dilution labels | 694 |
| Length proxy `D/(H+D)` vs segmented area dilution | r = 0.95, reads 3.8 points high on average |
| Depth model, R² random split / unseen settings / higher power | GPR 0.93 / 0.83 / 0.85, random forest 0.94 / 0.87 / 0.49 |
| Process-model transfer (residual GPR) | with 0–3 target conditions, depth is predicted better than by a from-scratch model trained on 12 |
| U-Net transfer to laser-wire IN718 (7 images) | clad IoU 0.42 → 0.93, dilution error 1.8 points |
| Coaxial camera | improves `H`, `W`; adds no information on `D` beyond the process settings |

## Data

The public dataset is described in

> J. Akhavan, J. Lyu, Y. Mahmoud, K. Xu, C. K. P. Vallabh, S. Manoochehri, "Dataset of in-situ coaxial
> monitoring and print's cross-section images by Direct Energy Deposition fabrication", *Scientific Data*
> 10, 776 (2023). https://doi.org/10.1038/s41597-023-02672-4

Download the list of file links from the dataset page given in the paper and save it as a text file
(one URL per line). The scripts expect this layout, relative to the project root:

```
data/optomec/Final_total_measurement.xlsx     expert H, W, D for every section
data/optomec/micro_tif/DED_SS316L_0001A.tif   cross-section micrographs (extracted)
data/optomec/coaxial/DED_SS316L_0001_denoise.zip   coaxial camera frames per track
```

The laser-wire DED micrographs are unpublished and not included.

## Installation

```
python -m pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.9 (CPU), scikit-learn 1.7, OpenCV 5.0.

## Running the pipeline

Run from the project root. Each script prints its usage in its docstring.

```
# 1. data
python pipeline/download.py links.txt data/optomec --only xlsx
python pipeline/download.py links.txt data/optomec --only microscope
python pipeline/download.py links.txt data/optomec --only denoise
python -I pipeline/zip_walk.py data/optomec/microscope_zips/<zip> data/optomec/micro_tif

# 2. rule-based segmentation + expert-agreement quality control
python pipeline/segment_optomec.py data/optomec/micro_tif data/optomec/Final_total_measurement.xlsx results/optomec_seg

# 3. self-trained U-Net
python pipeline/unet_seg.py data/optomec/micro_tif results/optomec_seg results/unet 40

# 4. process-to-geometry models under three validation protocols
python -I pipeline/process_model.py data/optomec/Final_total_measurement.xlsx results/optomec
python -I pipeline/dilution_targets.py results    # after step 3: needs the U-Net evaluation

# 5. few-shot transfer
python pipeline/transfer_process.py results/optomec/optomec_sections_mm.csv results/transfer_learning results
python pipeline/unet_transfer.py data/optomec/micro_tif results/optomec_seg results/transfer_learning

# 6. coaxial camera
python -I pipeline/coaxial_features.py data/optomec/coaxial results/optomec/optomec_sections_mm.csv results/optomec/coaxial_section_features.csv
python pipeline/coaxial_value.py results/optomec/optomec_sections_mm.csv results/optomec/coaxial_section_features.csv results/optomec

# 7. laser-wire DED (own micrographs)
python -I pipeline/segment_wire.py <panel_grid.png> results/wire
python pipeline/wire_unet_transfer.py results/wire/panels results/unet/unet.pt results/transfer_learning

# 8. figures
python pipeline/make_figures.py results figs
python -I pipeline/report_extra_figs.py . figs
python -I pipeline/explain_superellipse.py . figs/fig_seg_steps.png --en
python -I pipeline/explain_dp.py . figs/fig_dp.png --en
python -I pipeline/unet_transfer_examples.py data/optomec/micro_tif results/optomec_seg results/transfer_learning/unet_source.pt figs/fig_transfer_examples.png
python -I pipeline/coaxial_examples.py . figs/fig_coaxial_examples.png
```

## Scripts

| Script | Purpose |
|---|---|
| `download.py`, `zip_walk.py` | resumable parallel download; extraction of Deflate64 archives |
| `segment_optomec.py` | rule-based 316L segmentation (Otsu, RANSAC surface line, superellipse fit, dynamic-programming fusion line), expert-agreement QC, masks, overlays |
| `segment_wire.py` | three-phase segmentation of laser-wire IN718/SS304 micrographs |
| `unet_seg.py` | U-Net training on validated masks and evaluation against the experts |
| `process_model.py` | GPR / random forest / log-linear models; random, grouped and extrapolation validation |
| `dilution_targets.py` | the same models for the segmented area dilution of the validated sections |
| `transfer_process.py` | few-shot process-model transfer (source only, scratch, recalibration, Δ-learning) |
| `unet_transfer.py`, `wire_unet_transfer.py` | few-shot U-Net transfer (dark → bright mount; powder 316L → laser-wire IN718) |
| `coaxial_features.py`, `coaxial_value.py` | coaxial frame features, frame-to-section registration, value of the camera |
| `make_figures.py`, `report_extra_figs.py`, `explain_superellipse.py`, `explain_dp.py`, `unet_transfer_examples.py`, `coaxial_examples.py` | figures of the report |

## Citation

If you use this code, please cite the B.Tech project report:

> A. Agarwal, "Measuring and Predicting Penetration and Dilution in Directed Energy Deposition",
> B.Tech Project Report, Department of Mechanical Engineering, Indian Institute of Technology Guwahati, 2026.

and the dataset paper above.
