import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import load_results

try:
    import umap
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def main():
    if not UMAP_AVAILABLE:
        print("Error: umap-learn or scikit-learn not installed")
        return 1
    
    results_dir = Path(__file__).parent.parent / "results"
    
    print("=" * 60)
    print("AFFECTIVE STATE CLUSTERING ANALYSIS")
    print("=" * 60)
    
    results = load_results(results_dir / "stage2" / "mujoco")
    if results is None:
        print("No results found for stage2/mujoco")
        return 1
    
    all_affects = []
    for r in results[:10]:
        all_affects.append(r["affect_history"])
    
    affect_concat = np.vstack(all_affects)
    
    print(f"Total affect states: {affect_concat.shape[0]}")
    print(f"Affect dimensions: {affect_concat.shape[1]}")
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(affect_concat)
    
    print("\nClustering with k-means...")
    best_k = 3
    best_score = -1
    
    for k in range(3, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(affect_concat)
        score = silhouette_score(affect_concat, labels)
        print(f"k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"\nBest k: {best_k} (silhouette={best_score:.4f})")
    
    if best_score > 0.3:
        print("Clustering quality: GOOD")
        return 0
    elif best_score > 0.2:
        print("Clustering quality: MODERATE")
        return 0
    else:
        print("Clustering quality: POOR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
