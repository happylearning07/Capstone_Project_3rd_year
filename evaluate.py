# import torch
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from models.aae_model import AAE

# # 1. Load the Saved Data and Model
# checkpoint = torch.load('encoder_final.pth')
# test_data = np.load('test_data.npz')

# X_train, X_test = test_data['X_train'], test_data['X_test']
# y_train, y_test = test_data['y_train'], test_data['y_test']

# input_dim = checkpoint['input_dim']
# encoder = AAE(input_dim=input_dim).encoder
# encoder.load_state_dict(checkpoint['state_dict'])
# encoder.eval()

# # 2. Extract Latent Features (The "Paper" way)
# with torch.no_grad():
#     X_latent_train = encoder(torch.FloatTensor(X_train)).numpy()
#     X_latent_test = encoder(torch.FloatTensor(X_test)).numpy()

# # 3. Final Classification using Random Forest
# print("Training Final Classifier on Latent Features...")
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_latent_train, y_train)
# y_pred = clf.predict(X_latent_test)

# # 4. Results
# print("\n--- Final Project Results (AAE + Random Forest) ---")
# print(classification_report(y_test, y_pred))







import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from models.aae_model import AAE

# 1. Load the Saved Data and Model
print("Loading data...")
checkpoint = torch.load('encoder_final.pth')
test_data = np.load('test_data.npz')

X_train, X_test = test_data['X_train'], test_data['X_test']
y_train, y_test = test_data['y_train'], test_data['y_test']

input_dim = checkpoint['input_dim']
encoder = AAE(input_dim=input_dim).encoder
encoder.load_state_dict(checkpoint['state_dict'])
encoder.eval()

# 2. Extract Latent Features
print("Extracting Latent Features using AAE Encoder...")
with torch.no_grad():
    X_latent_train = encoder(torch.FloatTensor(X_train)).numpy()
    X_latent_test = encoder(torch.FloatTensor(X_test)).numpy()

# 3. Optimized Final Classification
print("Training Final Classifier (Optimized for Speed)...")
# n_jobs=-1 uses all CPU cores. max_samples=0.1 uses only 10% of data for training 
# (plenty for high accuracy with AAE features)
clf = RandomForestClassifier(
    n_estimators=50, 
    random_state=42, 
    n_jobs=-1, 
    max_samples=0.1, 
    verbose=1  # This will show you progress so you know it's not frozen
)

clf.fit(X_latent_train, y_train)
y_pred = clf.predict(X_latent_test)

# 4. Results
print("\n--- Final Project Results (AAE + Random Forest) ---")
print(classification_report(y_test, y_pred))