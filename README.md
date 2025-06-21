# Clustering: K-means and Gaussian Mixture Models

This Jupyter notebook demonstrates the implementation and application of clustering algorithms, specifically **K-means** and **Gaussian Mixture Models (GMMs)**, using the Iris dataset. The notebook includes both from-scratch implementations and comparisons with scikit-learn's versions. The structure is educational and step-by-step, with code, visualization, and explanations.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Task 1: Data Loading and Visualization](#task-1-data-loading-and-visualization)
3. [Task 2: Clustering Algorithms](#task-2-clustering-algorithms)
   - [2.1: K-means from Scratch](#task-21-k-means-from-scratch)
   - [2.2: K-means with scikit-learn](#task-22-k-means-with-scikit-learn)
   - [2.3: GMM with scikit-learn](#task-23-gmm-with-scikit-learn)
   - [2.4: GMM Posterior Visualization](#task-24-gmm-posterior-visualization)
4. [Iris Dataset Description](#iris-dataset-description)
5. [How to Use This Notebook](#how-to-use-this-notebook)
6. [Requirements](#requirements)

---

## Introduction

The notebook introduces clustering, focusing on **K-means** and **GMM** algorithms. It contrasts manual implementations with those provided by scikit-learn, aiming for clarity and intuition rather than maximum efficiency.

---

## Task 1: Data Loading and Visualization

**Purpose:**  
Load the Iris dataset from `.npy` files, display the structure of the data, and visualize it for an initial feel of clustering.

**Steps:**
- Load feature data and labels from `Iris_data.npy` and `Iris_labels.npy`.
- Print the raw data and label arrays to understand their structure.
- Visualize the dataset using a scatter plot of Sepal Length vs Sepal Width, colored by the true species label.

**Outcome:**  
Users gain intuition about the dataset’s separability and the natural groupings of the samples.

---

## Task 2: Clustering Algorithms

### Task 2.1: K-means from Scratch

**Purpose:**  
Implement the K-means algorithm without using machine learning libraries, to illustrate the mechanics of clustering.

**Steps:**
- Define a function to compute Euclidean distance between points and centroids.
- Implement the K-means algorithm:
  - Randomly initialize centroids.
  - Iteratively assign data points to the nearest centroid.
  - Update centroids as the mean of assigned points.
  - Repeat until convergence or a maximum number of iterations.
- Print progress at each iteration.
- Visualize both the predicted clusters and the actual labels for comparison.

**Outcome:**  
A clear, educational look at how K-means clustering works under the hood, with visual demonstration of its performance.

---

### Task 2.2: K-means with scikit-learn

**Purpose:**  
Use scikit-learn’s built-in `KMeans` class for clustering, and compare results to the manual implementation.

**Steps:**
- Import and initialize `KMeans` from scikit-learn.
- Fit the model to the data.
- Predict cluster labels.
- Visualize the clusters and their centroids on a scatter plot.
- Experiment with `n_init` and `init` (e.g., `k-means++`) for better initialization and results.

**Outcome:**  
Demonstrates the ease and power of library implementations, and validates the manual algorithm by comparing outputs.

---

### Task 2.3: GMM with scikit-learn

**Purpose:**  
Apply Gaussian Mixture Model clustering using scikit-learn’s `GaussianMixture` class.

**Steps:**
- Import and initialize `GaussianMixture`.
- Fit the model and predict cluster labels.
- Visualize the results, with means (centroids) marked on the scatter plot.

**Outcome:**  
Shows how GMM can find clusters with elliptical shapes or varying sizes, compared to K-means’ spherical clusters.

---

### Task 2.4: GMM Posterior Visualization

**Purpose:**  
Visualize the posterior probabilities (soft assignments) of each data point belonging to each Gaussian component.

**Steps:**
- Use `predict_proba` to get the probability (responsibility) of each cluster for each point.
- Create scatter plots for each component, colored by the posterior probability.
- Add color bars and legends for interpretation.

**Outcome:**  
Gives insight into the "soft" nature of GMM clustering, where each point can partially belong to multiple clusters.

---

## Iris Dataset Description

- **Samples:** 150 iris flowers.
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width (all numerical).
- **Labels:** Three classes (Setosa, Versicolor, Virginica), encoded as integers (0, 1, 2).

---

## How to Use This Notebook

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/nick2601/Clustering.git
   cd Clustering
   ```

2. **Ensure you have the data files:**  
   Place `Iris_data.npy` and `Iris_labels.npy` in the working directory.

3. **Install requirements:**  
   ```bash
   pip install numpy matplotlib pandas scikit-learn
   ```

4. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook 2377995_Clustering.ipynb
   ```

5. **Run the cells in order.**  
   Follow along with the explanations and visualizations.

---

## Requirements

- Python 3.x
- Jupyter Notebook
- numpy
- matplotlib
- pandas
- scikit-learn

---
