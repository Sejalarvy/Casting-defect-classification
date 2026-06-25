# Industrial Defect Detection — Casting Quality Inspection (CNN)

Classifying images of cast metal pump impellers as **OK** or **Defective**
using Convolutional Neural Networks, including a from-scratch CNN and a
transfer-learning model — automating a task currently done by manual
visual inspection on real manufacturing lines.

## Problem

This is one of the most common real-world applications of computer
vision in manufacturing: **automated visual quality inspection**. Casting
defects (rim chipping, burrs, surface marks) cause expensive order
rejections if they reach a customer, but manual inspection is slow and
inconsistent. A model that flags likely defects for human review can make
quality control faster and more reliable.

## Dataset

[Real-life industrial dataset of casting product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) —
512×512 grayscale top-view images of submersible pump impellers, collected
at an actual casting manufacturer, **Pilot Technocast** (Shapar, Rajkot,
India). Labeled `ok_front` (good) or `def_front` (defective) by their real
quality-control process. This project uses a representative subset
(~1,300 images of the full 7,348-image dataset), already split into
train/test folders.

## Approach

1. **Visual inspection first** — looked at real OK vs. defective images
   before modeling. Found that defects are often small, localized rim
   irregularities, not obvious whole-image differences — and confirmed
   directly that downscaling images too aggressively (128×128) visibly
   destroys this signal, which is why 224×224 was used instead.
2. **CNN built from scratch** — a compact 4-block convolutional network,
   trained with class weighting to correct for the ~60/40 class imbalance
   in the data.
3. **Honest evaluation** — reported precision/recall per class, not just
   overall accuracy, since the two classes have very different real-world
   costs when misclassified.
4. **Transfer learning improvement** — MobileNetV2 pretrained on
   ImageNet, used as a frozen feature extractor with a small trainable
   classifier head, to test whether the from-scratch model's ceiling was
   a data-size problem rather than an approach problem.

## Results

| Model | Test Accuracy | def_front recall | ok_front recall |
|---|---|---|---|
| CNN from scratch (224×224) | 73.8% | 0.89 | 0.50 |
| MobileNetV2 transfer learning | **93.1%** | 0.94 | 0.93 |

The from-scratch CNN is biased toward predicting "defective" when
uncertain — a defensible failure mode for quality control (a false alarm
costs a few seconds of re-inspection; a missed defect costs more), but a
real limitation. Transfer learning closes almost all of this gap and
produces balanced recall across both classes, confirming that reusing
ImageNet-pretrained visual features (edges, textures) compensates well
for the small size of this dataset subset.

## Repo Structure

```
casting-defect-detection/
├── data/
│   ├── train_data/{ok_front, def_front}/
│   └── test_data/{ok_front, def_front}/
├── notebook.ipynb       # full analysis, executed end to end
├── requirements.txt
└── README.md
```

## Running it

This notebook was developed and run on **Google Colab** (free T4 GPU) —
recommended, since the transfer-learning section downloads pretrained
ImageNet weights and benefits from GPU acceleration.

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Possible Extensions

- Train on the full 7,348-image dataset to see how much of the
  from-scratch CNN's gap closes with more data alone.
- Fine-tune (unfreeze) the later MobileNetV2 layers instead of only
  training the classifier head.
- Build a confidence-threshold system for routing uncertain predictions
  to human review, rather than a single hard cutoff.

## Reference

Dataset: ["Real-life industrial dataset of casting product"](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product),
courtesy of Pilot Technocast, Shapar, Rajkot.
