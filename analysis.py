import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob

csv_path = 'data/sample_labels.csv' 
image_folder = 'data/images'

df = pd.read_csv(csv_path)

print(f"Total Rows: {len(df)}")
print("\n--- Raw Data Example ---")
print(df[['Image Index', 'Finding Labels']].head(5))

all_labels = set()
for labels in df['Finding Labels']:
    for label in labels.split('|'):
        all_labels.add(label)

all_labels=sorted(list(all_labels))
print(f"\n--- Found {len(all_labels)} Unique Diseases ---")
print(all_labels)

for label in all_labels:
    df[label]=df['Finding Labels'].map(lambda result: 1.0 if label in result else 0.0)

print("\n--- Processed Data (Binary Columns) ---")
print(df[all_labels[5:10]].head(5))

label_counts=df[all_labels].sum().sort_values(ascending=False)

plt.figure(figsize=(10,8))
label_counts.plot(kind='bar')
plt.title('Distribution of Diseases in Sample Dataset')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.show()

all_image_paths = {os.path.basename(x): x for x in glob(os.path.join(image_folder, '*.png'))}

df['path'] = df['Image Index'].map(all_image_paths.get)

df = df.dropna(subset=['path'])
print(f"\nValid Images Found on Disk: {len(df)}")

df.to_csv('/Users/mayankraj/Desktop/My_Projects/ChestXray_Project/data/clean_labels.csv', index=False)
print("Saved processed labels to data/clean_labels.csv")