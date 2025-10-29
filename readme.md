# BMLB 2025-2026
The 2025-2026 version of the two-week Basic Machine Learning for Bioinformatics course taught at the University of Utrecht.

# Topics covered per day (lectures and practicals)
* Day 1: Linear regression, gradient descent, introduction to linear algebra
* Day 2: Logistic regression, regularisation, ROC curve, introduction to neural networks (NNs)
* Day 3: NN backpropagation algorithm, convolutional neural networks explained
* Day 4: NN backpropagation continued, K-means clustering, hierarchical clustering
* Day 5: Problems with high-dimensional data, Principal Component Analysis (PCA)
* Day 6: Working with scikit-learn, introduction to Keras, project introduction and start

# Dependencies and running the computer labs.

## Installing the `BMLB2025_2026` Conda environment

Follow these steps to create the course environment from `course_conda_environment.yaml` and make it available in JupyterLab.

---

## 1) Clone the course repository

Open a terminal and run:

```bash
git clone https://github.com/DieStok/BMLB2025_2026.git
cd BMLB2025_2026
```
If you don't know how to do this, check the [course reader](CourseReaderMLBasics2025_2026.pdf) (:

---

## 2) Prerequisites

- Install **Conda** (either [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution))

---

## 3) Create the environment

From the project folder (where `course_conda_environment.yaml` is located):

```bash
conda env create -f course_conda_environment.yaml
```

Activate the environment:

```bash
conda activate BMLB2025_2026
```

# *NOTE: you should always activate the environment before launching jupyter lab!*
In your terminal this should look like:

```bash
(course conda environment name) whatever_else_your_terminal_puts
#example for me:
(BMBL2025_2026) (ARM64) dstoker@MA014385 % 
```
---

## 4) Activate nbdev to remove merge conflicts when updating notebooks:

```bash
nbdev_install_hooks
```

---

## 5) Register the Jupyter kernel

This allows you to select the environment as a kernel inside JupyterLab or Jupyter Notebook:

```bash
python -m ipykernel install --user --name BMLB2025_2026 --display-name "Python (BMLB2025_2026)"
```

---

## 6) Launch JupyterLab

```bash
#make sure to activate conda beforehand!
jupyter lab
```

In the launcher, select the kernel **“Python (BMLB2025_2026)”**.


> **Note:** PyTorch installed via `pip` in this environment is **CPU-only** by default. GPU support is **not required** for this course.

---

## 7) Updating later (if the YAML changes)

From the same folder:

```bash
conda env update -n BMLB2025_2026 -f course_conda_environment.yaml --prune
```

---

# Pulling updates when I update materials during the course

1. Save ALL local changes (including untracked files) into a named stash
2. Get latest course changes
3. Reapply local work on top of the updates (so you don't lose your hard-earned code)

```bash
git stash push -u -m "my-lab-work-$(date +%F)"
git pull --rebase
git stash pop
```

# More information
For more information and resources, read the [course reader](CourseReaderMLBasics2025_2026.pdf).

# Words of thanks
Greatly inspired by/based on [Andrew Ng's course on Coursera](https://www.coursera.org/learn/machine-learning/home/welcome). The PCA part is based on [Prof. Victor Lavrenko's excellent lecture series](https://www.youtube.com/watch?v=IbE0tbjy6JQ&list=PLBv09BD7ez_5_yapAg86Od6JeeypkS4YM). Many thanks are owed to [Dr. Jeroen de Ridder](https://www.umcutrecht.nl/en/research/researchers/de-ridder-jeroen-j) for expert assistance. I thank [Dr. ir. Bas van Breukelen](https://www.uu.nl/staff/BvanBreukelen) for long-term assistance and [Prof. Dr. Berend Snel](https://tbb.bio.uu.nl/snel/group.html) for comments on the phylogenetics part. Any errors remain my own (and, with your help, will hopefully be noticed and rectified soon).