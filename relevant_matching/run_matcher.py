import sys
import os
import cv2
from feature_utils import build_feature_database, find_top_matches

def save_matched_images(input_path, match_paths, output_file="matched_result.jpg"):
    input_img = cv2.imread(input_path)
    input_img = cv2.resize(input_img, (224, 224))

    matched_imgs = [cv2.resize(cv2.imread(p), (224, 224)) for p in match_paths]
    all_imgs = [input_img] + matched_imgs

    combined = cv2.hconcat(all_imgs)
    cv2.imwrite(output_file, combined)
    print(f"Output saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_matcher.py <input_image_path>")
        sys.exit(1)

    input_img_path = sys.argv[1]

    if not os.path.exists("features.pkl"):
        build_feature_database(dataset_dir="dataset", output_file="features.pkl")

    top_matches = find_top_matches(input_img_path, top_n=5)

    print("\nTop Matching Images:")
    for path, score in top_matches:
        print(f"{path} — Score: {score:.4f}")

    save_matched_images(input_img_path, [path for path, _ in top_matches])
