# Volcanic Activity Prediction from Satellite Imagery

Binary classification of satellite image patches as **active** (smoke/ash present) or **dormant** volcanoes, using a ResNet-18 backbone fused with a classical Laplacian-of-Gaussian (LoG) smoke feature.

**Author:** Vinicius Quintana, St. Edward's University (Computer Vision course project)

## Approach

- **ResNet-18** (ImageNet-pretrained, fine-tuned) gives a 512-D visual feature vector.
- **LoG smoke score:** a 1-D feature from the Laplacian of a Gaussian-blurred grayscale patch. Sharp, irregular plume edges score high; smooth cloud edges score low.
- **Fusion head:** the 513-D vector goes through FC(256) → FC(64) → FC(1) with BatchNorm, Dropout and a sigmoid, giving P(active).
- **Training:** BCE loss, AdamW, cosine annealing, early stopping (patience 7). About 11.3M parameters; trained in about 10 minutes on a GPU.
- **Threshold:** the decision threshold τ = 0.488 is tuned on the **validation** set only. The test set is scored once.

## Results (600 held-out test patches, τ = 0.488)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 97.83% | 100.00% | 95.67% | 97.79% | 1.0000 |

Confusion matrix: TN 300, FP 0, FN 13, TP 287. A 7-view test-time augmentation run gave identical metrics, so the single-pass model is reported.

**Robustness checks**
- **Leakage audit (dHash):** a random validation split leaked (145 of 360 validation images were near-duplicates of training images). I rebuilt the split by scene groups (906 groups). There are now 0 near-duplicates across train, validation and test.
- **Out-of-distribution (V1 dataset, 200 images):** 99.50% accuracy, 99.00% recall.
- **Occlusion test:** masking the plume lowers mean P(active) from 0.993 to 0.671. Masking the background leaves it at 0.992.
- **Interpretability:** Grad-CAM and t-SNE plots.

**Benchmark context:** on the same dataset, the accuracy is 3rd among the published methods compared (Shakir et al. 2026; Amato et al. 2023). Those papers do not document their split or threshold selection, so this is context, not a controlled comparison.

## Limitations

- The 13 missed active patches are a threshold-transfer issue: ranking is perfect (AUC 1.0), but the validation set was easier than the test set.
- The 600 test patches are not 600 independent scenes (about 2.6 patches per scene), so the uncertainty on 97.83% is understated.
- There is no LoG ablation (ResNet-18 alone), so I cannot claim the LoG feature raises accuracy.

## Repository contents

- `Volcanic_Activity.ipynb`: the full notebook (data, model, training, evaluation, diagnostics, benchmark, results).
- `outputs/plots/`: saved figures.

The **model checkpoint and the dataset are not included in this repository.**

## Dataset

3,200 Sentinel-2 and Landsat-8 patches (224×224) of active and dormant volcanoes:

- **Volcanic_V2:** 3,000 patches (2,400 training pool + 600 test, balanced).
- **Volcanic (V1):** 200 patches, used only as an out-of-distribution test.

Sources:
- **Original source:** Amato, E.; Corradino, C.; Torrisi, F.; Del Negro, C. (2023). SqueezeNet_Dataset_TransferLearning. Zenodo (CC BY 4.0). https://doi.org/10.5281/zenodo.7944343
- **Copy used for this project:** "Volcanic Thermal Activity" on Kaggle (uploader: yasserhessein): https://www.kaggle.com/datasets/yasserhessein/volcanic-thermal-activity

Download the data and put the zip (or the extracted folder) in the working directory. The notebook unzips it and finds the `train/` and `test/` folders automatically (each has `YesActivity/` and `NoActivity/`). Check the license on each page before reusing the data.

## How to run

- **Notebook:** open it in Jupyter or Colab and run the cells in order. A GPU is recommended for training (T4 GPU in Colab).
- **Requirements:** `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pillow`.

## References

- Amato, E.; Corradino, C.; Torrisi, F.; Del Negro, C. (2023). A Deep Convolutional Neural Network for Detecting Volcanic Thermal Anomalies from Satellite Images. *Remote Sensing* 15(15), 3718. https://doi.org/10.3390/rs15153718
- Shakir et al. (2026). DCNNVA. *International Journal of Advances in Applied Sciences* 15(1), 281-292. https://doi.org/10.11591/ijaas.v15.i1.pp281-292

**Collaborators:** Claude and Google Gemini.
