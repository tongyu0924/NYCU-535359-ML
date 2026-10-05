# A1: Clustering and Outlier Detection

k-means and outlier detection on the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(classical, hip-hop, mandopop, metal; n = 3,408 tracks after de-duplication).

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
