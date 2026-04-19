"""
train_aae_classifier.py
------------------------
Trains a Random Forest classifier on the AAE encoder's latent space
and saves it as aae_classifier.pkl.

Mirrors evaluate.py exactly:
  - Applies SMOTE to X_train before fitting the classifier
  - Uses the same RF hyper-parameters as evaluate.py

Run this ONCE after training the AAE:
    python train_aae_classifier.py

Requirements:
    - encoder_final.pth   (produced by train_aae.py)
    - test_data.npz       (produced by train_aae.py)
"""

import torch
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from models.aae_model import AAE


print("Loading AAE encoder from encoder_final.pth ...")
checkpoint = torch.load("encoder_final.pth", map_location="cpu")
input_dim  = checkpoint["input_dim"]

encoder = AAE(input_dim=input_dim).encoder
encoder.load_state_dict(checkpoint["state_dict"])
encoder.eval()
print(f"  encoder ready  |  input_dim={input_dim}")


print("Loading test_data.npz ...")
data    = np.load("test_data.npz")
X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"]
X_test  = data["X_test"].astype(np.float32)
y_test  = data["y_test"]
print(f"  train: {X_train.shape}  |  test: {X_test.shape}")


print("Encoding training data into latent space ...")
with torch.no_grad():
    Z_train = encoder(torch.FloatTensor(X_train)).numpy()
print(f"  latent dim: {Z_train.shape[1]}")


print("Applying SMOTE to latent training features ...")
sm = SMOTE(random_state=42)
Z_train_res, y_train_res = sm.fit_resample(Z_train, y_train)
print(f"  after SMOTE: {Z_train_res.shape}  |  class counts: "
      f"{ {int(c): int(n) for c, n in zip(*np.unique(y_train_res, return_counts=True))} }")


print("Encoding test data into latent space ...")
with torch.no_grad():
    Z_test = encoder(torch.FloatTensor(X_test)).numpy()


print("Training Random Forest classifier ...")
clf = RandomForestClassifier(
    n_estimators = 50,          # matches evaluate.py
    n_jobs       = -1,
    random_state = 42,
)
clf.fit(Z_train_res, y_train_res)
print("  training done.")


print("\nEvaluation on held-out test set:")
y_pred = clf.predict(Z_test)
print(classification_report(y_test, y_pred, zero_division=0))


clf_path = "aae_classifier.pkl"
with open(clf_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Classifier saved -> {clf_path}")
print("You can now run:  python run_pipeline.py --mode stream --model aae --n 200")