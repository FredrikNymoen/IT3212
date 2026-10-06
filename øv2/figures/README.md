# Figures for the HOG and LBP report

Rendered by `render_report_figures.py` at the physical size they occupy in the
report, so LaTeX scales them by 1.0 and the text stays at 7 to 8.5 pt.
`\textwidth` is 455.24 pt (a4paper, 11pt, 2.5 cm margins). The widths below are
the ones used in `report_hog_lbp.tex`; where a figure is included narrower than
it was rendered, the text in it shrinks by the same factor.

| File | Included at | Used in the report | Note |
|---|---|---|---|
| hog_01_dataset.pdf | `\textwidth` | no | The five scenes on one row; they are also the top row of hog_03 |
| hog_02_gradients.pdf | `\textwidth` | no | $G_x$, $G_y$ and direction for the silhouette, the sedan and the pickup |
| hog_03_visualisation.pdf | `\textwidth` | yes | Image, gradient magnitude and HOG, one column per scene |
| hog_04_illumination.pdf | `\textwidth` | yes | |
| hog_05_orientation_profile.pdf | `\textwidth` | yes | |
| hog_06_cosine_similarity.pdf | `0.60\textwidth` | no | The values are quoted in the text |
| hog_07_parameters.pdf | `\textwidth` | no | Accuracy and cosine gap with bootstrap intervals; the report gives the same numbers as a table |
| hog_08_cell_size.pdf | `0.48\textwidth` | yes | Rendered for `0.70\textwidth` |
| hog_09_bins.pdf | `\textwidth` | yes | New: 4, 9 and 18 orientation bins |
| lbp_01_images.pdf | `0.72\textwidth` | yes | Rendered for `\textwidth` |
| lbp_02_histograms.pdf | `\textwidth` | yes | |
| lbp_03_chi2_matrix.pdf | `0.42\textwidth` | yes | Rendered for `0.60\textwidth` |
| lbp_04_noise.pdf | `\textwidth` | yes | |

All figures were regenerated after the image listing in the notebooks was fixed
(the old listing returned every file twice on Windows).

`lbp_01_overview.pdf` from the earlier export is superseded by
`lbp_01_images.pdf` and can be deleted.
