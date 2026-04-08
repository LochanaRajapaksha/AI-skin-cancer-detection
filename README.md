Skin Cancer SVM Classifier — `skin_cancer_svm_fast.pkl`
A binary image classifier that distinguishes benign from malignant skin lesions using a LinearSVC trained on flattened, scaled image features.
---
Model Overview
Property	Value
File	`skin_cancer_svm_fast.pkl`
Algorithm	LinearSVC (Support Vector Classifier)
Task	Binary classification
Classes	`benign` (0) · `malignant` (1)
Input features	240 (flattened/extracted image features)
Trained with	scikit-learn 1.6.1
---
Performance
Split	Accuracy
Training	93.2%
Validation	90.6%
Test	91.1%
---
File Contents
The `.pkl` file is a dictionary with four keys:
Key	Type	Description
`svm_model`	`LinearSVC`	Trained classifier
`scaler`	`StandardScaler`	Feature normalizer (fit on training data)
`pca`	`None`	PCA was not applied
`label_encoder`	`LabelEncoder`	Maps class names to integers
---
Model Hyperparameters
```
C              = 1.0
loss           = squared_hinge
penalty        = l2
multi_class    = ovr
dual           = False
fit_intercept  = True
max_iter       = 2000
random_state   = 42
tol            = 0.0001
```
---
Requirements
```
scikit-learn >= 1.6.1
numpy
joblib
```
Install with:
```bash
pip install scikit-learn joblib numpy
```
> **Note:** The model was saved with scikit-learn 1.6.1. Loading it with a newer version will work but may raise an `InconsistentVersionWarning`. For production use, match the scikit-learn version.
---
Usage
```python
import joblib
import numpy as np

# Load the model bundle
bundle = joblib.load("skin_cancer_svm_fast.pkl")

svm_model     = bundle["svm_model"]
scaler        = bundle["scaler"]
label_encoder = bundle["label_encoder"]

# Prepare your input — must be shape (n_samples, 240)
# Replace this with your actual feature extraction logic
features = np.array([...])  # shape: (n_samples, 240)

# Preprocess: scale features
features_scaled = scaler.transform(features)

# Predict
predictions = svm_model.predict(features_scaled)

# Decode labels
labels = label_encoder.inverse_transform(predictions)
print(labels)  # ['benign'] or ['malignant']
```
---
Input Format
Features must have exactly 240 dimensions per sample.
Apply the bundled `StandardScaler` before inference — do not re-fit it on new data.
If your pipeline extracts image features (e.g., flattened pixel values or HOG-style descriptors), ensure the same extraction method used during training is applied at inference.
---
Class Mapping
Integer	Label
0	benign
1	malignant
---
Limitations
The model does not use PCA (`pca` key is `None`), so all 240 raw features are used directly.
LinearSVC does not produce probability estimates by default. To get probabilities, wrap it with `CalibratedClassifierCV` during training.
Performance figures reflect the dataset used during training and may not generalise to significantly different image sources or acquisition conditions.
