# Casting Defect Detection (CNN)

A computer vision project that classifies cast metal pump impellers as **OK** or **Defective** from photos — basically teaching a model to do the visual quality check that's normally done by a person on a factory floor.

## Why I picked this

I wanted a project that actually used real images, not just spreadsheet data, and that connected to manufacturing or mechanical engineering rather than being a generic dataset everyone uses. Casting defect inspection fit that — it's a real problem factories deal with, and the dataset comes from an actual company, not a synthetic benchmark.

## The dataset

These are real inspection photos of submersible pump impellers from [Pilot Technocast](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), a casting manufacturer in Rajkot, India. Each image is labeled **ok_front** or **def_front** based on their actual quality control process.

Here's what the two classes actually look like:

![OK vs Defective samples](ok%20vs%20def.png)

Honestly, when I first looked at these, I couldn't always tell the difference myself — the defects are usually small chips or rough edges around the rim, not something obvious. That told me early on that this wasn't going to be an easy, near-perfect-accuracy problem unless I was careful about things like image resolution.

## What I did

**1. Looked at the data before touching any model.** I zoomed into a few images and noticed that if you shrink them too much, down to something like 128 by 128 pixels, the rim defects basically disappear. So I trained at 224 by 224 instead, which keeps that detail.

**2. Built a CNN from scratch first.** A simple 4-layer convolutional network. This got to about **74 percent test accuracy**. Not bad, but the model was clearly biased — it kept guessing "defective" more than it should have, especially when unsure.

**3. Tried transfer learning.** Instead of training a network from zero on about 1,300 images, which isn't a lot for image data, I used **MobileNetV2**, a network already trained on 1.4 million general images, and just trained a small classifier on top of its features. This jumped accuracy up to **93 percent**, with much more balanced results across both classes.

Confusion matrix for the better, transfer learning model:

![Confusion matrix](confusion%20matrix.png)

## Results

| Model | Test Accuracy |
|---|---|
| CNN trained from scratch | 73.8% |
| MobileNetV2 (transfer learning) | **93.1%** |

The jump from 74 to 93 percent is really the main finding here. It shows that with a small dataset, reusing features from a much bigger pretrained model can matter more than tweaking your own architecture.

## What I'd try next

- Train on the full dataset. This used a subset of about 1,300 images out of the original 7,348, so I'd want to see how much closer the from-scratch model gets with all of it.
- Unfreeze some of MobileNetV2's later layers and fine-tune them on this specific data instead of only training the top layer.
- Add a "send to human for review" step for predictions the model isn't confident about, instead of just a flat 50 percent cutoff.

## Running it

The notebook, **casting_defect_notebook.ipynb**, was built and run on Google Colab with a free GPU. You'll need to download the dataset from the Kaggle link above to run it yourself. It's not included in this repo since it's a few thousand images.

```bash
pip install -r requirements.txt
jupyter notebook casting_defect_notebook.ipynb
```
