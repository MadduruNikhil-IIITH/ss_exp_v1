import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

print("Loading features...")
data = pickle.load(open('features_cache.pkl', 'rb'))

rows = []
for split in ['train', 'validation']:
    for record in data[split]:
        row = {'binary_label': record['binary_label']}
        for k, v in record['features'].items():
            if k.startswith('rst_rel_'):
                row[k] = v
        rows.append(row)

df = pd.DataFrame(rows)

# Get the average counts for each relation, grouped by salience
grouped = df.groupby('binary_label').mean()

# Transpose for plotting
grouped = grouped.T
# Rename columns
grouped.columns = ['Non-Salient', 'Salient']
# Rename index
grouped.index = [x.replace('rst_rel_', '').replace('_count', '').capitalize() for x in grouped.index]

print("Generating plot...")
plt.figure(figsize=(8, 5))
grouped.plot(kind='bar', figsize=(8, 5), color=['#3498db', '#f1c40f'], edgecolor='black', alpha=0.8)
plt.title("Distribution of RST Relations: Salient vs. Non-Salient", fontsize=12, fontweight='bold')
plt.ylabel("Average Count per Sentence", fontsize=10)
plt.xlabel("RST Relation Type", fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, linestyle=":", alpha=0.6, axis="y")
plt.tight_layout()

os.makedirs(os.path.join("docs", "images"), exist_ok=True)
out_path = os.path.join("docs", "images", "rst_relations_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
