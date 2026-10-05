# A1: Clustering and Outlier Detection

k-means and outlier detection on the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(classical, hip-hop, mandopop, metal; n = 3,408 tracks after de-duplication).

## Assignment overview

The assignment studies unsupervised learning in three parts: hand-worked exercises,
k-means implemented from scratch, and a comparison of outlier detectors on real music data.

**Part I: Clustering and PCA by hand**
- Why PCA must centre the data before computing principal components
- k-means on a small 1-D example: how different initialisations reach different local optima, and how farthest-first initialisation reacts to an outlier
- Single-link vs. complete-link hierarchical clustering and the chaining effect
- DBSCAN core / border / noise points, compared with k-means

**Part II: k-means from scratch**
- Data preparation: keep four genres, remove duplicate tracks, standardise nine audio features
  (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo)
- Implement vectorised k-means in NumPy and verify it against scikit-learn
- Compare random, farthest-first and k-means++ initialisation over 20 seeds
- Choose k (2–8) with the elbow plot, silhouette score and ARI against genre labels,
  and test whether the cluster structure is real using column-permuted null datasets

**Part III: Outlier detection**
- z-score masking and Local Outlier Factor by hand
- Implement kNN-distance and LOF scores from scratch; inspect the most unusual tracks
- Compare with a clustering-based score, DBSCAN noise points and Isolation Forest
- Inject synthetic global and correlation-breaking anomalies and evaluate every detector
  with ROC AUC, precision@20, and precision / recall / F1

## Contents

| File | Description |
|---|---|
| `A1_315511079.pdf` | Report (Part I hand calculations, results, analysis) |
| `Part2.ipynb` | k-means from scratch, initialisation strategies, choosing k, null test |
| `Part3.ipynb` | kNN distance and LOF from scratch, DBSCAN, Isolation Forest, injected-anomaly evaluation |
| `ML-part2-figure/`, `ML-part3-figure/` | Exported figures and result tables |

## Key results

- Custom k-means matches scikit-learn (difference 3.6e-12); best SSE for k = 4 is about 15,811
- Silhouette favours k = 2 and ARI against genre favours k = 4; the real SSE is well below the permuted-null SSE
- Custom LOF matches scikit-learn to 1.1e-10; LOF20 is the best detector on both global (AUC 0.999) and correlation-breaking (AUC 0.889) anomalies

## Run

Open the notebooks in Google Colab and run all cells. The data are read directly with
`pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")`. Random seeds are fixed.
