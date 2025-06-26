# feature_utils.py

import os
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

# === Load pre-trained ResNet-50 (remove last classification layer) ===
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# === Define image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Extract 2048-d feature vector from one image ===
def extract_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
        return features.flatten()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to extract features from {img_path}: {e}")

# === Extract features from dataset and save to .pkl ===
def build_feature_database(dataset_dir='dataset', output_file='features.pkl'):
    feature_list = []
    image_paths = []

    print("🔍 Extracting features from dataset images...\n")

    for file in tqdm(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, file)

        # ✅ Skip folders and non-image files
        if not os.path.isfile(path) or not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"⏭️ Skipped (not an image): {file}")
            continue

        try:
            print(f"📷 Processing: {file}")
            features = extract_features(path)
            feature_list.append(features)
            image_paths.append(path)
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n✅ Total valid images processed: {len(feature_list)}")

    if len(feature_list) == 0:
        raise RuntimeError("❌ No valid dataset images found. Cannot build feature database.")

    with open(output_file, 'wb') as f:
        pickle.dump((feature_list, image_paths), f)

    print(f"✅ Feature database saved to: {output_file}\n")

# === Find top N matches by cosine similarity ===
def find_top_matches(input_img_path, top_n=5, feature_file='features.pkl'):
    print(f"\n🔍 Matching input image: {input_img_path}")

    input_features = extract_features(input_img_path).reshape(1, -1)

    with open(feature_file, 'rb') as f:
        feature_list, image_paths = pickle.load(f)

    if not feature_list:
        raise ValueError("❌ Dataset features are empty. Cannot match.")

    feature_array = np.array(feature_list)
    similarities = cosine_similarity(input_features, feature_array)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]

    top_matches = [(image_paths[i], similarities[i]) for i in top_indices]
    return top_matches