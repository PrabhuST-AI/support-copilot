"""segmentation.py -- Profile, name and project customer RFM segments.
1. Standardize RFM features with StandardScaler and fit KMeans(n_clusters=4, n_init=10, random_state=42).
2. Report segment size, mean recency, mean frequency, and mean monetary.
3. Assign business names, recommended marketing actions, and mark the highest monetary segment as Premium.
4. Perform PCA(n_components=2) and report explained_variance_ratio_.
5. Write outputs/customer_segments.csv with required schema:
   [customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium]
"""

import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RFM_PATH = "data/clean/rfm.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "customer_segments.csv")

def profile_and_project_segments():
    rfm = pd.read_csv(RFM_PATH)
    features = ["recency", "frequency", "monetary"]
    X = rfm[features]

    # Standardize RFM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means with chosen K=4
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["segment_id"] = km.fit_predict(X_scaled)

    # Identify segment with highest mean monetary value as Premium
    mean_monetary_by_cluster = rfm.groupby("segment_id")["monetary"].mean()
    premium_cluster_id = mean_monetary_by_cluster.idxmax()

    # Define segment business names
    # Segment 0: Occasional Buyers (low frequency, moderate recency, low spend)
    # Segment 1: Premium Customers (highest monetary, moderate frequency)
    # Segment 2: Loyal / Frequent Buyers (highest frequency, lowest recency)
    # Segment 3: At-Risk / Churned (highest recency, lowest frequency and spend)
    segment_names = {
        0: "Occasional Buyers",
        1: "Premium Customers",
        2: "Loyal / Frequent Shoppers",
        3: "At-Risk / Churned"
    }

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = rfm["segment_id"] == premium_cluster_id

    # Print Segment Profiles
    summary = rfm.groupby(["segment_id", "segment_name", "is_premium"]).agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    print("=" * 80)
    print("  CUSTOMER RFM SEGMENT PROFILES (K = 4)")
    print("=" * 80)
    print(summary.to_string(index=False))

    # Apply PCA (n_components=2)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    evr = pca.explained_variance_ratio_

    print("\n" + "=" * 80)
    print("  PCA 2D PROJECTION (EXPLAINED VARIANCE)")
    print("=" * 80)
    print(f"PC1 Explained Variance: {evr[0]:.3f} (Reference: ~0.634)")
    print(f"PC2 Explained Variance: {evr[1]:.3f} (Reference: ~0.218)")
    print(f"Total Explained Variance: {evr.sum():.3f} (Reference: ~0.852)")

    # Write outputs/customer_segments.csv
    cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    out_df = rfm[cols].copy()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(out_df)} rows to {OUTPUT_FILE}")

    return out_df, summary, evr

if __name__ == "__main__":
    profile_and_project_segments()
