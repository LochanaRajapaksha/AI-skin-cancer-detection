"""
Skin Cancer SVM Classifier
===========================
Binary classifier: benign vs malignant skin lesions.
Model: LinearSVC | Features: 240 | Saved with scikit-learn 1.6.1
"""

import joblib
import numpy as np


# ── Load model bundle ──────────────────────────────────────────────────────────

def load_model(path="skin_cancer_svm_fast.pkl"):
    bundle = joblib.load(path)
    return {
        "svm":     bundle["svm_model"],
        "scaler":  bundle["scaler"],
        "encoder": bundle["label_encoder"],
    }


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(image):
    """
    Convert an image to a 240-dimensional feature vector.
    Replace this with the same method used during training.

    Args:
        image: numpy array (H, W) or (H, W, C)

    Returns:
        numpy array of shape (240,)
    """
    from PIL import Image as PILImage

    # Resize to a fixed size whose flattened length matches 240
    # e.g. grayscale 15x16 = 240, or adjust to your actual pipeline
    img = PILImage.fromarray(image).convert("L").resize((16, 15))
    features = np.array(img).flatten().astype(np.float32)
    assert features.shape == (240,), f"Expected 240 features, got {features.shape[0]}"
    return features


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(features, model):
    """
    Run inference on one or more samples.

    Args:
        features: numpy array of shape (n_samples, 240)
        model:    dict returned by load_model()

    Returns:
        labels: list of strings, e.g. ['benign', 'malignant']
    """
    features = np.atleast_2d(features)
    scaled = model["scaler"].transform(features)
    preds = model["svm"].predict(scaled)
    labels = model["encoder"].inverse_transform(preds)
    return labels.tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model = load_model("skin_cancer_svm_fast.pkl")
    print("Model loaded successfully.")
    print(f"  Classes : {model['encoder'].classes_.tolist()}")
    print(f"  Features: {model['svm'].n_features_in_}")
    print(f"  Kernel  : LinearSVC")

    # ── Example: predict from an image file ───────────────────────────────────
    if len(sys.argv) > 1:
        from PIL import Image
        img_path = sys.argv[1]
        img = np.array(Image.open(img_path))
        features = extract_features(img)
        result = predict(features, model)
        print(f"\nImage : {img_path}")
        print(f"Result: {result[0].upper()}")

    # ── Example: predict from random dummy data ───────────────────────────────
    else:
        print("\nNo image path provided — running on random dummy data.")
        dummy = np.random.rand(3, 240).astype(np.float32)
        results = predict(dummy, model)
        for i, label in enumerate(results):
            print(f"  Sample {i + 1}: {label}")
