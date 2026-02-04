import pandas as pd
import os

# 1. Load the Labels
df = pd.read_csv('data/clean_labels.csv')

# 2. Filter for Cardiomegaly
# We look for rows where Cardiomegaly == 1
disease = 'Cardiomegaly'
sick_patients = df[df[disease] == 1]

print(f"Found {len(sick_patients)} patients with {disease}")

# 3. Check which files actually exist on disk (Sanity Check)
# Sometimes the CSV has more rows than downloaded images
valid_images = []
for index, row in sick_patients.iterrows():
    img_path = row['path']
    if os.path.exists(img_path):
        valid_images.append(img_path)
        if len(valid_images) >= 10: 
            break

print("\n--- TEST THESE IMAGES ---")
for img in valid_images:
    print(f"Path: {img}")