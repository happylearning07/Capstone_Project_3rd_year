"""
train_aae.py  (fixed)
---------------------
Key fix: saves FULL AAE state_dict (encoder + decoder + discriminator)
as 'aae_final.pth' IN ADDITION to the encoder-only 'encoder_final.pth'.

Why this matters:
  The adversarial (FGSM) mode in gateway.py needs a trained decoder to
  compute reconstruction error as the loss signal.  The original code only
  saved encoder weights, so decoder weights were random at adversarial time,
  making FGSM ineffective for the AAE model.

Backward compatibility: encoder_final.pth is still saved identically,
so train_aae_classifier.py and evaluate.py are unaffected.

CRITICAL: n_rows MUST match train_bigan.py (500_000).
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.aae_model import AAE
from utils.preprocessing import load_and_clean, apply_smote
import numpy as np

N_ROWS = 500_000

X_train, X_test, y_train, y_test, input_dim = load_and_clean(
    'data/iot23_combined_new.csv', n_rows=N_ROWS)

X_res, y_res = apply_smote(X_train, y_train)

print(f"input_dim : {input_dim}  (should be 39 when n_rows={N_ROWS})")
print(f"X_train   : {X_train.shape}")
print(f"X_res     : {X_res.shape}  (after SMOTE)")

dataset = TensorDataset(torch.FloatTensor(X_res))
loader  = DataLoader(dataset, batch_size=128, shuffle=True)

latent_dim = 16
model      = AAE(input_dim=input_dim, latent_dim=latent_dim)
recon_loss = torch.nn.MSELoss()
dc_loss    = torch.nn.BCELoss()

optim_ae = optim.Adam(
    list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-4)
optim_dc = optim.Adam(model.discriminator.parameters(), lr=1e-4)
optim_en = optim.Adam(model.encoder.parameters(), lr=1e-4)

for epoch in range(50):
    for (batch_x,) in loader:
        batch_size = batch_x.size(0)

        # Phase 1: Reconstruction
        optim_ae.zero_grad()
        z     = model.encoder(batch_x)
        x_hat = model.decoder(z)
        loss_recon = recon_loss(x_hat, batch_x)
        loss_recon.backward()
        optim_ae.step()

        # Phase 2a: Discriminator
        z_real = torch.randn(batch_size, latent_dim)
        z_fake = model.encoder(batch_x)

        optim_dc.zero_grad()
        d_loss = (dc_loss(model.discriminator(z_real), torch.ones(batch_size, 1)) +
                  dc_loss(model.discriminator(z_fake.detach()), torch.zeros(batch_size, 1)))
        d_loss.backward()
        optim_dc.step()

        # Phase 2b: Encoder fools discriminator
        optim_en.zero_grad()
        g_loss = dc_loss(model.discriminator(z_fake), torch.ones(batch_size, 1))
        g_loss.backward()
        optim_en.step()

    print(f"Epoch {epoch} | Recon Loss: {loss_recon.item():.4f}")

torch.save(
    {'state_dict': model.encoder.state_dict(), 'input_dim': input_dim},
    'encoder_final.pth'
)

torch.save(
    {'state_dict': model.state_dict(), 'input_dim': input_dim,
     'latent_dim': latent_dim},
    'aae_final.pth'
)

np.savez('test_data.npz',
         X_test=X_test,   y_test=y_test,
         X_train=X_train, y_train=y_train)

print(f"\nSaved encoder_final.pth  (input_dim={input_dim})  - encoder only")
print(f"Saved aae_final.pth      (input_dim={input_dim})  - FULL model (enc+dec+disc)")
print("Next: python train_aae_classifier.py")