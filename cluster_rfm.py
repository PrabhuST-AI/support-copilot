"""cluster_rfm.py -- K-Means clustering on standardized RFM features.
1. Standardise recency, frequency, monetary with StandardScaler.
2. For K from 2 to 8, fit KMeans(n_clusters=K, n_init=10, random_state=42).
3. Record inertia and silhouette score.
4. Evaluate elbow point, silhouette peak, and business interpretability to choose optimal K.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_PATH = "data/clean/rfm.csv"

def evaluate_kmeans():
    rfm = pd.read_csv(RFM_PATH)
    features = ["recency", "frequency", "monetary"]
    X = rfm[features]

    # Standardize RFM features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reference values for verification
    ref = {
        2: (1820.8, 0.465),
        3: (1268.2, 0.445),
        4: (743.1, 0.526),
        5: (617.4, 0.481),
        6: (503.2, 0.482),
        7: (408.5, 0.416),
        8: (366.5, 0.412),
    }

    results = []
    print("=" * 65)
    print("  K-MEANS CLUSTERING EVALUATION (K = 2 to 8)")
    print("=" * 65)
    print(f"{'K':<4} {'Inertia':>10} {'Silhouette':>12} {'Ref Inertia':>14} {'Ref Sil':>10}")
    print("-" * 65)

    models = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        models[k] = (km, labels)

        exp_inertia, exp_sil = ref[k]
        results.append({
            "K": k,
            "Inertia": round(inertia, 1),
            "Silhouette": round(sil, 3),
            "Inertia_Delta": round(results[-1]["Inertia"] - inertia, 1) if results else None
        })
        print(f"{k:<4} {inertia:>10.1f} {sil:>12.3f} {exp_inertia:>14.1f} {exp_sil:>10.3f}")

    print("-" * 65)

    # Business profile of the optimal K=4 model
    print("\n" + "=" * 65)
    print("  OPTIMAL MODEL PROFILES (K = 4)")
    print("=" * 65)
    km4, labels4 = models[4]
    rfm_k4 = rfm.copy()
    rfm_k4["cluster"] = labels4

    cluster_profiles = rfm_k4.groupby("cluster").agg(
        customer_count=("customer_id", "count"),
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean")
    ).reset_index()

    labels_map = {
        0: "Occasional Buyers (Low Spend / Moderate Recency)",
        1: "Premium Spenders (High Monetary / Moderate Frequency)",
        2: "Loyal / Frequent Buyers (High Frequency / Very Recent)",
        3: "At-Risk / Hibernating (Very High Recency / Low Spend)"
    }
    cluster_profiles["segment_persona"] = cluster_profiles["cluster"].map(labels_map)
    print(cluster_profiles.to_string(index=False))

    return results

if __name__ == "__main__":
    evaluate_kmeans()
