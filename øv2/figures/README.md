# Figures for the HOG and LBP report

Rendered by `render_report_figures.py` at the physical size they occupy in the
report, so LaTeX scales them by 1.0 and the text stays at 7 to 8.5 pt.
`\textwidth` is 455.24 pt (a4paper, 11pt, 2.5 cm margins). Use exactly the
width given below, otherwise the font sizes drift.

| File | Include at | Replaces | Note |
|---|---|---|---|
| hog_01_dataset.pdf | `\textwidth` | (new) | Companion to Table 1, optional |
| hog_02_gradients.pdf | `\textwidth` | f16_hog_gradients | Magnitude column dropped, it is column 2 of hog_03 |
| hog_03_visualisation.pdf | `0.74\textwidth` | f17_hog_visualisation | |
| hog_04_illumination.pdf | `\textwidth` | f20_hog_illumination | |
| hog_05_orientation_profile.pdf | `\textwidth` | (new) | |
| hog_06_cosine_similarity.pdf | `0.60\textwidth` | (new) | |
| hog_07_parameters.pdf | `\textwidth` | f21_hog_parameters | 200 images, 40 per class |
| hog_08_cell_size.pdf | `0.70\textwidth` | f19_hog_cellsize | 2x2 instead of 1x5, original panel dropped |
| lbp_01_images.pdf | `\textwidth` | f22_lbp_images | Histogram row moved to lbp_02 |
| lbp_02_histograms.pdf | `\textwidth` | f23_lbp_histograms | Four linear histograms plus the log overlay |
| lbp_03_chi2_matrix.pdf | `0.60\textwidth` | (new) | |
| lbp_04_noise.pdf | `\textwidth` | f24_lbp_noise | |

`lbp_01_overview.pdf` from the earlier export is superseded by
`lbp_01_images.pdf` and can be deleted.

Not covered here, because the notebooks print those results as tables rather
than plotting them: f34_hog_ablation, f35_letterbox, f36_lbp_smoothing,
f37_lbp_variants, f38_fusion.
