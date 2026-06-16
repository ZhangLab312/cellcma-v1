"""
LLM Code Generator

Dynamically generates code using LLM with automatic error correction support.
"""

from __future__ import annotations

import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

_TASK_INSTRUCTIONS = {
    "integration": """
## Task: Multi-Omics Integration
Integrate scRNA-seq and scATAC-seq data into a shared latent space where cells of the same type cluster together regardless of modality.

## CRITICAL: Strategy Selection Based on Data Pairing
Check the data context for "Paired data:" field to determine which strategy to use.
- If "Paired data: True" → Use **Strategy A** (Cross-Modal Prediction + Contrastive)
- If "Paired data: False" → Use **Strategy B** (DANN + Contrastive, NO cross-modal prediction)

**IMPORTANT**: Using the wrong strategy will cause FAILED integration:
- Strategy A on unpaired data: cross-modal prediction loss is noise → modalities remain separated
- Strategy B on paired data: DANN is overkill, wastes the cell correspondence signal

### Why both strategies work:
- VAE learns meaningful latent representations per modality
- Cross-modal contrastive loss (InfoNCE) forces RNA and ATAC cells of the same type to be nearby
- Cell type classifier auxiliary head improves cell type separation in latent space
- KL warmup prevents latent collapse at the start of training
- For PAIRED: Cross-modal prediction decoder uses known cell correspondence for alignment
- For UNPAIRED: Domain-Adversarial Discriminator (DANN) forces modality-invariant representations

### Architecture Details (FOLLOW EXACTLY):

#### Encoder (shared architecture, separate instances for RNA and ATAC):
```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.1)])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)
```
- RNA encoder: hidden_dims=[512, 256], latent_dim=128
- ATAC encoder: hidden_dims=[512, 256, 128], latent_dim=128 (DEEPER network to match RNA encoder capacity)

#### Cell Type Classifier (auxiliary head, CRITICAL for improving ASW_celltype):
```python
class CellTypeClassifier(nn.Module):
    def __init__(self, latent_dim, n_cell_types):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_cell_types),
        )
    def forward(self, z):
        return self.classifier(z)
```
- Applied to latent z from BOTH RNA and ATAC encoders
- Uses CrossEntropyLoss with cell type labels

#### Strategy A Component — Cross-Modal Prediction Decoder (ONLY for PAIRED data):
```python
class CrossModalDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, z_source):
        return self.predictor(z_source)
```
- RNA latent -> predict ATAC latent, and vice versa
- Enforces cross-modal alignment by forcing one modality's latent to predict the other's
- **ONLY use for paired data** — for unpaired data this is meaningless noise

#### Strategy B Component — Domain-Adversarial Discriminator (ONLY for UNPAIRED data):
```python
class GradientReversalLayer(torch.autograd.Function):
    \"\"\"Gradient Reversal Layer: identity in forward, negate in backward.\"\"\"
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class ModalityDiscriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z, alpha=1.0):
        z_rev = GradientReversalLayer.apply(z, alpha)
        return self.net(z_rev)
```
- The discriminator tries to predict modality (RNA=0, ATAC=1) from latent z
- The Gradient Reversal Layer negates gradients flowing back to the encoder
- This forces the encoder to learn modality-invariant representations
- Alpha schedule: `alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0` where p = epoch/epochs
- **ONLY use for unpaired data** — for paired data, use cross-modal prediction instead

#### Loss Functions — Strategy A (PAIRED data):
1. **Reconstruction loss** (MSE on PCA/LSI features): weight=1.0
2. **KL divergence** (standard VAE): warmup 0→0.005 over first 50 epochs
   ```python
   kl_weight = min(0.005, epoch / 50.0 * 0.005)
   ```
3. **Cross-modal contrastive loss** (InfoNCE with hard negative mining): weight=2.5
   - Use matching cell pairs as positives (same barcode)
   - temperature=0.3
4. **MMD alignment** (multi-scale RBF): weight=0.5, bandwidths=[1.0, 2.0, 5.0, 10.0, 20.0]
5. **Cell type classifier** (CrossEntropy): weight=0.5
6. **Cross-modal prediction** (MSE): weight=0.3

Total loss (paired) = recon + kl_weight*KL + 2.5*contrastive + 0.5*MMD + 0.5*classifier + 0.3*cross_modal_pred

#### Loss Functions — Strategy B (UNPAIRED data):
1. **Reconstruction loss** (MSE on PCA/LSI features): weight=1.0
2. **KL divergence**: warmup 0→0.005 over first 50 epochs
3. **InfoNCE contrastive loss** with cell-type positive pairs: weight=3.0
   - Same cell type = positive pair, different cell type = negative
   - temperature=0.5 (higher than paired because cell-type positives are noisier than matched pairs)
   - Hard negative mining: top-K hardest negatives per anchor
4. **MMD alignment** (multi-scale RBF): weight=1.0 (higher than paired — primary alignment mechanism)
   - bandwidths=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] (more bandwidths for better coverage)
5. **Cell type classifier** (CrossEntropy): weight=1.0 (higher — critical for unpaired)
6. **DANN adversarial loss** (BCEWithLogitsLoss): weight=1.0
   - Concatenate RNA and ATAC latents: z_all = cat([z_rna, z_atac])
   - Labels: 0 for RNA cells, 1 for ATAC cells
   - discriminator_pred = discriminator(z_all, alpha=dann_alpha)
   - dann_loss = BCEWithLogitsLoss(discriminator_pred.squeeze(), modality_labels)
   - Alpha ramp-up: `dann_alpha = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0`

Total loss (unpaired) = recon + kl_weight*KL + 3.0*contrastive + 1.0*MMD + 1.0*classifier + 1.0*dann

#### CRITICAL Training Protocol (MUST FOLLOW):
```python
# Training parameters
latent_dim = 128  # Increased from 64 for better capacity
lr = 1e-3
batch_size = 512  # Increased from 256 for better contrastive learning
epochs = 500  # MINIMUM 500 epochs, use 800 if dataset < 5000 cells
weight_decay = 1e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Learning rate warmup: linear warmup for first 50 epochs, then cosine decay
warmup_epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

def get_lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

# Early stopping with patience
patience = 50
best_loss = float('inf')
patience_counter = 0

# Gradient clipping
max_grad_norm = 1.0

n_rna = X_rna.shape[0]
n_atac = X_atac.shape[0]

for epoch in range(epochs):
    model.train()
    # KL warmup
    kl_weight = min(0.005, epoch / 50.0 * 0.005)

    # Apply warmup scheduler for first 50 epochs, then cosine
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    elif epoch == warmup_epochs:
        # Switch to cosine scheduler
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    else:
        scheduler.step()

    rna_perm = torch.randperm(n_rna)
    atac_perm = torch.randperm(n_atac)
    n_batches = max(n_rna, n_atac) // batch_size + 1

    epoch_loss = 0.0
    for b in range(n_batches):
        rna_start = (b * batch_size) % n_rna
        rna_idx = rna_perm[rna_start:rna_start + batch_size]
        atac_start = (b * batch_size) % n_atac
        atac_idx = atac_perm[atac_start:atac_start + batch_size]

        rna_batch = X_rna_tensor[rna_idx]
        atac_batch = X_atac_tensor[atac_idx]
        rna_labels_batch = rna_cell_type_indices[rna_idx]
        atac_labels_batch = atac_cell_type_indices[atac_idx]

        # Forward pass, compute all losses, backward, step
        ...
        # CRITICAL: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        epoch_loss += loss.item()

    # Early stopping check
    avg_loss = epoch_loss / n_batches
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - KL weight: {kl_weight:.6f}")
```

#### Cross-Modal Contrastive Loss with Hard Negative Mining:
```python
def contrastive_loss(z_rna, z_atac, rna_labels, atac_labels, temperature=0.3, hard_neg_ratio=0.3):
    # Normalize embeddings
    z_rna_norm = nn.functional.normalize(z_rna, dim=1)
    z_atac_norm = nn.functional.normalize(z_atac, dim=1)

    # Compute similarity matrix (RNA x ATAC)
    sim = torch.mm(z_rna_norm, z_atac_norm.t()) / temperature

    # Create positive mask: same cell type = positive pair
    max_cls = max(rna_labels.max(), atac_labels.max()) + 1
    label_sim = torch.mm(
        nn.functional.one_hot(rna_labels, num_classes=max_cls).float(),
        nn.functional.one_hot(atac_labels, num_classes=max_cls).float().t()
    )
    positive_mask = (label_sim > 0).float()

    # Hard negative mining: mask out easy negatives (low similarity)
    neg_mask = (1 - positive_mask)
    n_keep = max(1, int(neg_mask.sum(dim=1).float().mean() * hard_neg_ratio))
    neg_sim = sim * neg_mask - 1e4 * positive_mask
    _, hard_indices = neg_sim.topk(min(n_keep, neg_sim.shape[1]), dim=1)
    hard_neg_mask = torch.zeros_like(sim)
    hard_neg_mask.scatter_(1, hard_indices, 1.0)
    final_mask = positive_mask + hard_neg_mask

    n_positives = positive_mask.sum(dim=1).clamp(min=1)
    log_prob = sim - torch.logsumexp(sim * final_mask + (1 - final_mask) * (-1e4), dim=1, keepdim=True)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / n_positives
    loss = -mean_log_prob_pos.mean()
    return loss
```

#### DANN Loss for Unpaired Data (Strategy B ONLY):
```python
# Inside training loop, after computing z_rna and z_atac:
dann_alpha = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0

z_all = torch.cat([mu_rna, mu_atac], dim=0)
mod_labels = torch.cat([torch.zeros(mu_rna.size(0)), torch.ones(mu_atac.size(0))]).to(device)
d_pred = discriminator(z_all, alpha=dann_alpha)
dann_loss = nn.BCEWithLogitsLoss()(d_pred.squeeze(), mod_labels)
# Add to total_loss with weight=1.0
# The GRL ensures encoder learns to FOOL the discriminator (modality-invariant)
```

### Preprocessing (CRITICAL - DO NOT SKIP):
1. **RNA**: normalize_total (1e4) → log1p → HVG selection (top 2000) → PCA(50)
   ```python
   # Sparse-safe normalization
   if sp.issparse(X):
       lib = np.asarray(X.sum(axis=1)).ravel()
       lib[lib == 0] = 1.0
       X = sparse.diags(1e4 / lib) @ X
       X.data = np.log1p(X.data)
   else:
       lib = X.sum(axis=1, keepdims=True)
       lib[lib == 0] = 1.0
       X = np.log1p(X / lib * 1e4)
   # Then PCA to 50 components
   ```
2. **ATAC**: TF-IDF → LSI(50)
   ```python
   # TF-IDF
   tf = X / X.sum(axis=1)  # term frequency
   idf = np.log(1 + n_cells / (1 + (X > 0).sum(axis=0)))  # inverse document frequency
   X_tfidf = tf * idf
   # Then TruncatedSVD to 50 components (LSI)
   ```
3. **Standardize** both PCA and LSI outputs (zero mean, unit variance) before feeding to the model

### STABILITY CONSTRAINTS (MUST FOLLOW - prevent training collapse):
1. ALWAYS clamp logvar: `torch.clamp(logvar, -10, 10)`
2. ALWAYS use KL warmup: start from 0, ramp to 0.005 over 50 epochs
3. ALWAYS use gradient clipping: `clip_grad_norm_(params, max_norm=1.0)`
4. ALWAYS check for NaN/Inf in loss: `if torch.isnan(loss) or torch.isinf(loss): continue`
5. NEVER use learning rate > 2e-3
6. ALWAYS use BatchNorm1d in encoder layers (stabilizes training)
7. If latent variance drops below 0.01 after 50 epochs, reduce KL weight to 0.001

## Evaluation Metrics (REQUIRED)

### Joint NMI and ARI (ALWAYS compute this way — cluster ALL cells together)
This is the STANDARD evaluation for multi-omics integration. Cluster ALL cells (RNA + ATAC) jointly in the shared latent space, then compare with ground truth cell type labels:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

# Concatenate latent representations: RNA cells first, then ATAC cells
latent_all = np.concatenate([latent_rna, latent_atac], axis=0)  # shape: (n_rna + n_atac, latent_dim)
cell_type_all = np.concatenate([rna_cell_types, atac_cell_types])  # string labels for all cells
modality_all = np.concatenate([np.zeros(n_rna), np.ones(n_atac)])  # 0=RNA, 1=ATAC

# Encode cell type labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cell_type_encoded = le.fit_transform(cell_type_all)

# Cluster ALL cells jointly
n_clusters = len(np.unique(cell_type_all))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
pred_labels = kmeans.fit_predict(latent_all)

# Compute NMI and ARI on ALL cells
nmi = normalized_mutual_info_score(cell_type_encoded, pred_labels)
ari = adjusted_rand_score(cell_type_encoded, pred_labels)
```
Required output keys: `metrics.nmi`, `metrics.ari`

### Per-modality NMI/ARI (ALSO compute, for diagnostic purposes)
```python
rna_mask = modality_all == 0
atac_mask = modality_all == 1
# Use the SAME KMeans model fitted on all cells
nmi_rna = normalized_mutual_info_score(cell_type_encoded[rna_mask], pred_labels[rna_mask])
ari_rna = adjusted_rand_score(cell_type_encoded[rna_mask], pred_labels[rna_mask])
nmi_atac = normalized_mutual_info_score(cell_type_encoded[atac_mask], pred_labels[atac_mask])
ari_atac = adjusted_rand_score(cell_type_encoded[atac_mask], pred_labels[atac_mask])
```
Required output keys: `metrics.nmi_rna`, `metrics.ari_rna`, `metrics.nmi_atac`, `metrics.ari_atac`

### ASW metrics (ALWAYS compute)
```python
sample_size = min(5000, len(cell_type_all))
# ASW cell type: how well cell types are separated — HIGHER IS BETTER
asw_celltype = silhouette_score(latent_all, cell_type_encoded, sample_size=sample_size)
# ASW batch (modality mixing): how well RNA and ATAC cells are mixed
asw_batch = 1 - abs(silhouette_score(latent_all, modality_all, sample_size=sample_size))
```
Required output keys: `metrics.asw_celltype`, `metrics.asw_batch`

### kNN cross-modality accuracy
```python
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=min(50, len(latent_all)-1)).fit(latent_all)
_, indices = nn.kneighbors(latent_all)
knn_cross = np.mean([np.mean(modality_all[indices[i]] != modality_all[i]) for i in range(len(modality_all))])
```
Required output key: `metrics.knn_cross`

**Result format:**
```python
result = {
    "status": "success",
    "latent_representation": latent_all,  # REQUIRED: numpy (n_total_cells, latent_dim)
    "predicted_labels": pred_labels,
    "metrics": {
        "nmi": float(nmi),
        "ari": float(ari),
        "nmi_rna": float(nmi_rna),
        "ari_rna": float(ari_rna),
        "nmi_atac": float(nmi_atac),
        "ari_atac": float(ari_atac),
        "asw_celltype": float(asw_celltype),
        "asw_batch": float(asw_batch),
        "knn_cross": float(knn_cross),
    }
}

# Store embeddings in adata for downstream tasks
n_rna = adata_rna.shape[0]
adata_rna.obsm['X_integration'] = latent_all[:n_rna]
adata_atac.obsm['X_integration'] = latent_all[n_rna:]
```

**CRITICAL**: `latent_representation` MUST be a numpy array. The system saves it as h5ad.

**Note**: If ground truth labels are unavailable, set metric to `None` but include the key.
""",
    "grn": """
## Task: Multi-Omics Gene Regulatory Network Inference
Infer gene regulatory networks by jointly leveraging scRNA-seq (expression) and scATAC-seq (chromatin accessibility / TF binding) data.

Requirements:
1. Use RNA expression to compute gene-gene co-expression relationships
2. Use ATAC chromatin accessibility to identify active regulatory elements and TF binding sites
3. Link distal ATAC peaks to target genes (peak-to-gene linkage)
4. Integrate both modalities to score regulatory edges (TF → target gene)
5. Apply thresholding to get sparse network
6. Save the edge list as a CSV file and return it in the result

Key considerations:
- ATAC data has extremely high dimensionality (hundreds of thousands of peaks). ALWAYS reduce dimensionality first (e.g., LSI/TruncatedSVD to 50 components). NEVER do `adata_atac.X.toarray()`.
- Use TF motif information from ATAC peaks to identify candidate regulators
- Cross-reference ATAC peak accessibility with RNA expression of nearby genes
- Handle sparse data appropriately
- Normalize expression values before computing correlations

**Prior network** (if `prior_network` is provided in data_info):
- Load the prior network file and use it as known TF-target relationships to guide inference
- Prior edges should receive boosted confidence scores

## Output Format (REQUIRED)
You MUST output the predicted regulatory network as a CSV file with exactly 3 columns:

| TF | Target | EdgeWeight |
|---|---|---|
| GENE1 | GENE2 | 0.85 |
| GENE1 | GENE3 | 0.72 |

- `TF`: transcription factor gene name (must match gene names in the data)
- `Target`: target gene name
- `EdgeWeight`: confidence score for the regulatory edge (higher = more confident)
- Sort by EdgeWeight in descending order
- Include ALL predicted edges (do not filter too aggressively)
- Save to: `outputs/grn_edge_list.csv`

**Result format:**
```python
import os
os.makedirs("outputs", exist_ok=True)
edge_df.to_csv("outputs/grn_edge_list.csv", index=False)

result = {
    "status": "success",
    "edge_list_file": "outputs/grn_edge_list.csv",
    "n_edges": len(edge_df),
    "n_tfs": edge_df["TF"].nunique(),
    "metrics": {}
}
```

**Note**: Evaluation metrics will be computed externally using BEELINE benchmark. You do NOT need to compute them yourself.
""",
    "perturbation": """
## Task: Perturbation Response Prediction
Predict single-cell gene expression responses under perturbations (e.g., drug treatment, chemical exposure, cytokine stimulation, environmental intervention) using control cells as baseline and perturbation identity / dosage / context as conditioning signals.

Requirements:
1. Identify control vs perturbed cells from `adata.obs`
2. Detect perturbation labels, and if available, dose / time / batch / cell type covariates
3. Build a predictive model that maps baseline cellular state + perturbation condition -> perturbed gene expression
4. Predict expression changes for held-out perturbation cells or unseen combinations when possible
5. Return predicted expression, true expression, perturbation labels, log fold-change summaries, and all required metrics

Key considerations:
- Control cells are the baseline reference and MUST be used explicitly
- Perturbation labels may be stored under columns such as `perturbation`, `condition`, `treatment`, `drug`, `compound`, etc. Your code should detect these robustly
- If dosage / concentration / time information exists, use it as an additional conditioning variable
- Cell type heterogeneity is critical: responses may differ strongly across cell states, so the model should condition on baseline expression or cell identity
- Single-cell matrices are often sparse and noisy; use normalization and optionally HVG selection before modeling
- The method should work even if only a subset of perturbations has many cells

## Recommended Data Handling (IMPORTANT)
Before training, you SHOULD preprocess data carefully:

1. Expression preprocessing:
   - If counts are raw, use library-size normalization + log1p
   - If already normalized, avoid double normalization
   - Use highly variable genes (e.g., top 1000-3000 HVGs) when gene dimension is very large
   - Keep a mapping back to original genes if output on full genes is required

2. Covariate detection:
   - Detect perturbation column from `adata.obs`
   - Detect control cells from:
     - `adata.obs['control'] == True`, OR
     - perturbation labels such as `control`, `ctrl`, `vehicle`, `dmso`, `untreated`, `baseline`
   - Detect optional columns:
     - cell type: `cell_type`, `celltype`, `cluster`, `label`
     - dose: `dose`, `dosage`, `concentration`
     - time: `time`, `timepoint`
     - batch: `batch`

3. Train/test protocol:
   - If the dataset already has train/test split metadata, follow it
   - Otherwise, evaluate on held-out perturbed cells
   - If enough perturbation categories exist, prefer perturbation-aware splitting to avoid information leakage

## Practical Modeling Tips
- Train 100-300 epochs depending on model size
- Use GPU for all tensor operations
- If perturbation classes are highly imbalanced, use balanced sampling or per-perturbation minibatches
- If there are many genes, predict only HVGs and optionally project back
- Evaluate both whole-transcriptome fidelity and DE-focused fidelity

## Modeling Objective (IMPORTANT)
Your prediction target should ideally include :
    **Change-aware objective**
   - Encourage correct perturbation effect relative to matched control
   - Example:
     delta_true = x_true - x_control_ref
     delta_pred = x_pred - x_control_ref
     loss_delta = MSE(delta_pred, delta_true)

**Multi-Metric Optimization (CRITICAL — read carefully):**
Your loss function MUST optimize R2, MSE, and PCC simultaneously. A common failure mode is achieving good R2/PCC but poor MSE (or vice versa).

Why this happens: R2 and PCC are scale-invariant (they measure correlation/shape), but MSE penalizes absolute magnitude errors. A model can have high R2/PCC yet poor MSE if predictions are systematically biased or scaled incorrectly.

Required loss design:
```python
# Primary: MSE directly minimizes absolute error
loss_mse = nn.MSELoss()(pred, target)

# Auxiliary: Pearson correlation loss (differentiable approximation)
# This ensures the predicted shape/trend matches the true expression
def pearson_loss(pred, target):
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    cos_sim = nn.functional.cosine_similarity(pred_centered, target_centered, dim=0)
    return 1.0 - cos_sim.mean()

loss_corr = pearson_loss(pred, target)

# DE-gene weighted loss: higher weight on differentially expressed genes
# de_mask is a boolean tensor of shape (n_genes,) marking DE genes
loss_de = nn.MSELoss()(pred[:, de_mask], target[:, de_mask])

# Total loss: combine all objectives
loss = 1.0 * loss_mse + 0.5 * loss_corr + 0.5 * loss_de
```

IMPORTANT constraints:
- Do NOT use tanh/sigmoid output activation if target values exceed [-1, 1]. Delta expression values can have arbitrary magnitude — use linear output or adaptive scaling.
- If using a VAE/CVAE architecture, ensure the reconstruction loss is MSE-based (not BCE), because BCE assumes [0,1] range which delta expression does not satisfy.
- Always include a plain MSE term in the total loss — this is the ONLY way to directly optimize the MSE metric.

## Evaluation Metrics (REQUIRED)
After perturbation prediction, you MUST compute and include these metrics in the `result` dictionary.

### 1. R2 (Coefficient of Determination)
Overall prediction accuracy on the full expression matrix.
- Formula:
  R2 = 1 - Sum(x_ij - xhat_ij)^2 / Sum(x_ij - xbar_j)^2
- Use:
  `from sklearn.metrics import r2_score`
- Required output key:
  `metrics.r2`

### 2. MSE (Mean Squared Error)
Absolute prediction error on the full expression matrix.
- Formula:
  MSE = mean((x_ij - xhat_ij)^2)
- Use:
  `from sklearn.metrics import mean_squared_error`
- Required output key:
  `metrics.mse`

### 3. PCC (Pearson Correlation Coefficient — overall)
Overall linear correlation between predicted and true expression profiles.
- Compute:
  `pcc, _ = pearsonr(y_true.ravel(), y_pred.ravel())`
- This measures whether predicted and true values have consistent absolute magnitude and trend.
- Use:
  `from scipy.stats import pearsonr`
- Required output key:
  `metrics.pcc`

### 4. DE-focused metrics on Top-20 Differentially Expressed Genes
You MUST identify top 20 DE genes using perturbed vs control comparison, then recompute evaluation on these genes only.

DE gene selection:
- For each gene, compare perturbed vs control using t-test or another simple differential test
- Rank genes by absolute t-statistic (or absolute logFC if necessary)
- Select top 20 genes
- Required output key:
  `de_genes`

Then compute:

#### 4a. R2_DE
- R2 on predicted vs true expression restricted to the selected 20 DE genes
- Required output key:
  `metrics.r2_de`

#### 4b. MSE_DE
- MSE on predicted vs true expression restricted to the selected 20 DE genes
- Required output key:
  `metrics.mse_de`

#### 4c. PCC_DE
- Pearson correlation of predicted vs true fold-change restricted to the selected 20 DE genes
- Same procedure as PCC, but only on DE genes
- Required output key:
  `metrics.pcc_de`

## Data structure assumptions
- `adata`: AnnData object with:
  - `.X`: gene expression matrix
  - `.obs`: metadata table
- Expected / possible metadata columns:
  - perturbation identity:
    - `.obs['perturbation']` or similar detected column
  - control flag:
    - `.obs['control']` or inferred from label names
  - optional:
    - `.obs['cell_type']`
    - `.obs['dose']`
    - `.obs['time']`
    - `.obs['batch']`

**Result format example:**
```python
result = {
    "status": "success",
    "predicted_expression": predicted_matrix,  # shape (n_cells, n_genes)
    "true_expression": true_matrix,            # shape (n_cells, n_genes)
    "control_mask": control_bool_array,          # boolean array for control cells
    "perturbation_labels": pert_labels,          # perturbation conditions
    "de_genes": de_gene_indices,                # indices of top 20 DE genes
    "foldchange_predicted": fc_pred,              # predicted fold changes
    "foldchange_true": fc_true,                   # true fold changes
    "metrics": {
        "r2": 0.75,
        "mse": 0.12,
        "pcc": 0.85,
        "r2_de": 0.82,
        "mse_de": 0.08,
        "pcc_de": 0.79
    }
}
```

**Note**: Compute all metrics on single-cell level, then average. If ground truth is not available, set metrics to `None`.
""",
    "genetic_perturbation": """
## Task: Genetic Perturbation Response Prediction
Predict single-cell gene expression responses under genetic perturbations, including knockout (KO), knockdown (KD), overexpression (OE), CRISPR perturbation, guide-RNA perturbation, and combinatorial gene perturbation.

Requirements:
1. Identify wild-type / non-perturbed control cells vs genetically perturbed cells
2. Detect perturbation targets and perturbation types
3. Build a predictive model that maps basal cell state + genetic perturbation -> perturbed gene expression
4. Predict expression responses for held-out perturbed cells and, when possible, unseen perturbation targets or combinations
5. Return predicted expression, true expression, perturbation labels, DE genes, log fold-change summaries, and all required metrics

Key considerations:
- Genetic perturbation is not the same as generic treatment perturbation: the identity of the targeted gene(s) is biologically meaningful and should be modeled explicitly
- WT / non-targeting / empty-vector / scramble / control guide cells should be treated as baseline controls
- Your code should parse target gene identity when possible
- Cell state strongly modulates genetic perturbation response, so conditioning on baseline expression is essential
- If guide-level information exists (gRNA / sgRNA / guide_id), it can be used as a lower-level perturbation representation
- Combinatorial perturbations (e.g., double KO) may appear and should be handled if present

## Recommended Data Handling (IMPORTANT)

1. Expression preprocessing:
   - If counts are raw, apply library normalization + log1p
   - Use HVGs if gene dimension is very large
   - Preserve mapping to original genes

2. Perturbation parsing:
   - Detect perturbation column from `.obs`, e.g.:
     `perturbation`, `condition`, `gene_perturbation`, `target`, `guide`, `gRNA`
   - Infer control cells from:
     - `.obs['control'] == True`, OR
     - labels such as `WT`, `CTRL`, `control`, `NTC`, `non_targeting`, `scramble`, `empty_vector`
   - If perturbation label strings contain gene + mode, parse:
     - target gene(s)
     - perturbation type (KO/KD/OE/CRISPRi/CRISPRa)
   - Detect optional columns:
     - cell type
     - batch
     - guide / sgRNA
     - multiplicity / number of targets
     - timepoint

3. Splitting / evaluation:
   - If dataset provides train/test split, use it
   - Otherwise evaluate on held-out perturbed cells
   - If many perturbation targets exist, a stronger protocol is target-wise holdout:
     train on some genes, test on unseen perturbed genes
   - Avoid leakage across guides targeting the same gene if performing gene-level generalization


## Practical Modeling Tips
- If gene target identity is available, learn a perturbation embedding table
- Train 100-300 epochs depending on dataset and model size
- Use GPU for tensor operations
- If perturbation targets are imbalanced, use balanced sampling across target genes
- If there are multiple guides per gene, consider guide-level averaging or hierarchical encoding
- Evaluate both transcriptome-wide accuracy and DE-focused response accuracy

## Modeling Objective (IMPORTANT)
Your loss should capture both absolute expression prediction and perturbation effect prediction.

1. **Effect / delta loss**
   - Compare predicted perturbation effect relative to WT/control reference:
     delta_true = x_true - x_control_ref
     delta_pred = x_pred - x_control_ref
     loss_delta = MSE(delta_pred, delta_true)

**Multi-Metric Optimization (CRITICAL):**
Same as the perturbation task: your loss MUST optimize R2, MSE, and PCC simultaneously.
Include: (1) MSE loss for absolute error, (2) Pearson correlation loss for shape/trend, (3) DE-gene weighted loss.
Do NOT use tanh/sigmoid output activation for delta expression. Always include a plain MSE term in total loss.

# Evaluation Metrics (REQUIRED) — same output keys as perturbation task
All metrics use the same output structure as the generic perturbation task, but here each perturbation refers to a genetic perturbation condition.

### 1. R2 (Coefficient of Determination)
Overall prediction accuracy on the full expression matrix.
- Formula:
  R2 = 1 - Sum(x_ij - xhat_ij)^2 / Sum(x_ij - xbar_j)^2
- Use:
  `from sklearn.metrics import r2_score`
- Required output key:
  `metrics.r2`

### 2. MSE (Mean Squared Error)
Absolute prediction error on the full expression matrix.
- Formula:
  MSE = mean((x_ij - xhat_ij)^2)
- Use:
  `from sklearn.metrics import mean_squared_error`
- Required output key:
  `metrics.mse`

### 3. PCC (Pearson Correlation Coefficient — overall)
Overall linear correlation between predicted and true expression profiles.
- Compute:
  `pcc, _ = pearsonr(y_true.ravel(), y_pred.ravel())`
- Use:
  `from scipy.stats import pearsonr`
- Required output key:
  `metrics.pcc`

### 4. DE-focused metrics on Top-20 Differentially Expressed Genes
First identify DE genes by comparing perturbed vs WT/control cells:
- use t-test or similar simple statistical test
- rank genes by absolute t-statistic
- select top 20 genes
- Required output key:
  `de_genes`

Then compute:

#### 4a. R2_DE
- R2 restricted to the selected 20 DE genes
- Required output key:
  `metrics.r2_de`

#### 4b. MSE_DE
- MSE restricted to the selected 20 DE genes
- Required output key:
  `metrics.mse_de`

#### 4c. PCC_DE
- Pearson correlation restricted to the selected 20 DE genes
- Required output key:
  `metrics.pcc_de`

## Data structure assumptions
- `adata`: AnnData object with:
  - `.X`: gene expression matrix
  - `.obs`: metadata table
- Expected / possible columns:
  - perturbation label:
    - `.obs['perturbation']` or detected equivalent
  - control / WT indicator:
    - `.obs['control']` or inferred from labels
  - optional:
    - `.obs['cell_type']`
    - `.obs['batch']`
    - `.obs['guide']` / `.obs['gRNA']`
    - `.obs['target_gene']`
    - `.obs['perturbation_type']`

**Result format example:**
```python
result = {
    "status": "success",
    "predicted_expression": predicted_matrix,  # shape (n_cells, n_genes)
    "true_expression": true_matrix,            # shape (n_cells, n_genes)
    "control_mask": control_bool_array,          # boolean array for WT/non-perturbed cells
    "perturbation_labels": pert_labels,          # perturbation conditions
    "de_genes": de_gene_indices,                # indices of top 20 DE genes
    "foldchange_predicted": fc_pred,              # predicted fold changes
    "foldchange_true": fc_true,                   # true fold changes
    "metrics": {
        "r2": 0.75,
        "mse": 0.12,
        "pcc": 0.85,
        "r2_de": 0.82,
        "mse_de": 0.08,
        "pcc_de": 0.79
    }
}
```

**Note**: Compute all metrics on single-cell level, then average. If ground truth is not available, set metrics to `None`.
""",
}


_RESPONSE_FORMAT_BLOCK = r"""
## Response Format (STRICT - READ CAREFULLY)

You MUST output a JSON line FOLLOWED by a Python code block.

### Output Structure
1. One JSON line with minimal metadata:
{{"reasoning": "brief", "dependencies": [], "estimated_complexity": "low"}}

2. IMMEDIATELY after, a ```python code block.

### CRITICAL: Inline Comments Only (No Prose Outside Code)

**ALL explanations MUST be inline comments inside the code block.**

**WRONG:**
```text
The model uses a VAE architecture with MMD loss for alignment.
We train for 100 epochs with batch size 256.

```python
import torch
...
```

**CORRECT:**
```python
# Approach: VAE + MMD loss for RNA-ATAC alignment (100 epochs, batch 256)
import torch
...
```

Keep external explanation to 1 line max. The JSON "reasoning" field is enough.

### CRITICAL - Pipeline Priority (Prevents Truncation)

Generate your code in ORDER OF IMPORTANCE so important parts appear early:

1. `nn = torch.nn` (first line, always)
2. Imports
3. ALL class definitions
4. `result = {{"status": "placeholder"}}` ← PUT THIS EARLY as a scaffold
5. Data preprocessing
6. Model instantiation + GPU setup
7. Training loop
8. `result = {{...}}` with real metrics ← overwrite the placeholder

**Why this matters:** If the response is truncated, having `result = {{...}}` early ensures
you always have a return value. The parser will extract whatever is complete up to that point.

### Code Completeness Requirements

- The LAST non-empty line of your code MUST be `result = {{...}}` containing all metrics
- Include: model definition, training, prediction, AND metrics computation
- DO NOT stop at a class definition or function definition
- The `result` dictionary is MANDATORY

### Correct Output Example
{{"reasoning": "VAE+MMD for integration", "dependencies": ["torch"], "estimated_complexity": "medium"}}
```python
# Approach: VAE with MMD alignment, 100 epochs
nn = torch.nn
import torch

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        # ... full implementation ...

result = {{"status": "placeholder"}}  # scaffold early, fill later
# ... training loop ...
result = {{"status": "success", "nmi": nmi_score, "ari": ari_score}}
```

### Wrong Patterns
- WRONG: Large paragraphs of explanation outside the code block
- WRONG: {{"code": "..."}} — code inside JSON string
- WRONG: Only defining a class without training or returning result
- WRONG: `result = {{...}}` buried deep in the code instead of early
"""

_DIMENSION_PROBE_CODE = r'''
# ===== AUTO-GENERATED: DIMENSION PROBE =====
# Probes injected data dimensions and makes them available as global variables
# so downstream model code cannot hardcode wrong dimensions. DO NOT modify.

import numpy as np

_GLOBAL_N_CELLS = None
_GLOBAL_N_GENES = None
_GLOBAL_CELL_TYPES = None
_GLOBAL_N_CELL_TYPES = None
_GLOBAL_BATCHES = None
_GLOBAL_N_BATCHES = None
_GLOBAL_PERTURBATIONS = None
_GLOBAL_N_PERTURBATIONS = None
_GLOBAL_RNA_N_CELLS = None
_GLOBAL_RNA_N_GENES = None
_GLOBAL_ATAC_N_CELLS = None
_GLOBAL_ATAC_N_PEAKS = None

def _probe_data():
    global _GLOBAL_N_CELLS, _GLOBAL_N_GENES, _GLOBAL_CELL_TYPES, _GLOBAL_N_CELL_TYPES
    global _GLOBAL_BATCHES, _GLOBAL_N_BATCHES, _GLOBAL_PERTURBATIONS, _GLOBAL_N_PERTURBATIONS
    global _GLOBAL_RNA_N_CELLS, _GLOBAL_RNA_N_GENES, _GLOBAL_ATAC_N_CELLS, _GLOBAL_ATAC_N_PEAKS
    adata = globals().get('adata')
    adata_rna = globals().get('adata_rna')
    adata_atac = globals().get('adata_atac')

    if adata_rna is not None:
        _GLOBAL_RNA_N_CELLS, _GLOBAL_RNA_N_GENES = adata_rna.shape
        print(f"[DIM] RNA: n_cells={_GLOBAL_RNA_N_CELLS}, n_genes={_GLOBAL_RNA_N_GENES}")

    if adata_atac is not None:
        _GLOBAL_ATAC_N_CELLS, _GLOBAL_ATAC_N_PEAKS = adata_atac.shape
        print(f"[DIM] ATAC: n_cells={_GLOBAL_ATAC_N_CELLS}, n_peaks={_GLOBAL_ATAC_N_PEAKS}")

    target = adata_rna or adata or adata_atac
    if target is not None:
        _GLOBAL_N_CELLS, _GLOBAL_N_GENES = target.shape
        print(f"[DIM] Primary: n_cells={_GLOBAL_N_CELLS}, n_features={_GLOBAL_N_GENES}")
        if hasattr(target.obs, 'cell_type') and 'cell_type' in target.obs.columns:
            _GLOBAL_CELL_TYPES = target.obs['cell_type'].values
            _GLOBAL_N_CELL_TYPES = len(np.unique(_GLOBAL_CELL_TYPES))
            print(f"[DIM] n_cell_types={_GLOBAL_N_CELL_TYPES}, cell_types={list(np.unique(_GLOBAL_CELL_TYPES)[:10])}")
        if hasattr(target.obs, 'batch') and 'batch' in target.obs.columns:
            _GLOBAL_BATCHES = target.obs['batch'].values
            _GLOBAL_N_BATCHES = len(np.unique(_GLOBAL_BATCHES))
            print(f"[DIM] n_batches={_GLOBAL_N_BATCHES}")
        if hasattr(target.obs, 'perturbation') and 'perturbation' in target.obs.columns:
            _GLOBAL_PERTURBATIONS = target.obs['perturbation'].values
            _GLOBAL_N_PERTURBATIONS = len(np.unique(_GLOBAL_PERTURBATIONS))
            print(f"[DIM] n_perturbations={_GLOBAL_N_PERTURBATIONS}")
        if adata_rna is not None and hasattr(adata_rna.obs, 'cell_type') and 'cell_type' in adata_rna.obs.columns:
            _GLOBAL_CELL_TYPES = adata_rna.obs['cell_type'].values
            _GLOBAL_N_CELL_TYPES = len(np.unique(_GLOBAL_CELL_TYPES))
            print(f"[DIM] (rna) n_cell_types={_GLOBAL_N_CELL_TYPES}")
    else:
        print("[DIM] WARNING: No AnnData found, using fallback dims")
        _GLOBAL_N_GENES = 1000
        _GLOBAL_N_CELLS = 100
        _GLOBAL_N_CELL_TYPES = 5

_probe_data()
del _probe_data
# ===== END DIMENSION PROBE =====
'''


def _validate_and_return_static(
    result: "CodeGenerationResult", strategy_name: str, probe_code: str
) -> "CodeGenerationResult":
    """
    Module-level code return function. Syntax checking and error correction are entirely
    performed by LLM at runtime. No static compile check is done to reduce unnecessary
    rejections and automatic fixes.
    """
    logger_static = logging.getLogger(__name__)
    logger_static.info(f"[{strategy_name}] Code length: {len(result.code)} chars, {len(result.code.splitlines())} lines")

    try:
        debug_dir = Path("/root/autodl-tmp/BioAgent/debug")
        debug_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = debug_dir / f"{strategy_name}_{timestamp}.py"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result.code)
    except Exception:
        pass

    result.code = probe_code + result.code
    logger_static.info(f"[{strategy_name}] After probe prepend: {len(result.code)} chars")
    return result


logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationResult:
    """Code generation result"""
    code: str
    reasoning: str
    dependencies: List[str]
    estimated_complexity: str


class LLMCodeGenerator:
    """
    LLM Code Generator

    Uses LLM to dynamically generate code, supporting:
    1. Generate code based on task description and data information
    2. Fix code based on error messages
    3. Iteratively optimize code
    """

    def __init__(self, llm_config: Optional[Any] = None):
        self.llm_config = llm_config
        self.llm_client = None
        self.generation_history: List[Dict[str, Any]] = []
        self._fix_attempt_count: Dict[str, int] = {}

    def reset_fix_count(self):
        self._fix_attempt_count.clear()

    def reset_history(self):
        """Clear generation history to free memory. Called at the start of each round."""
        self.generation_history.clear()

    async def __aenter__(self) -> "LLMCodeGenerator":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the LLM client's aiohttp session to free network resources"""
        if self.llm_client is not None:
            try:
                await self.llm_client._reset_session()
            except Exception:
                pass
            self.llm_client = None

    def _get_llm_client(self):
        """Get LLM client"""
        if self.llm_client is None:
            try:
                from Agent.llm.llm_client import LLMClient, LLMConfig
                if self.llm_config is None:
                    self.llm_config = LLMConfig.from_env()
                self.llm_client = LLMClient(self.llm_config)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise
        return self.llm_client

    async def generate_code(
        self,
        task_type: str,
        data_info: Dict[str, Any],
        blueprint: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
        memory_context: str = "",
    ) -> CodeGenerationResult:
        """
        Generate code.

        Args:
            task_type: Task type
            data_info: Data information
            blueprint: Architecture blueprint (optional)
            iteration: Current iteration number
            memory_context: Memory context

        Returns:
            CodeGenerationResult: Code generation result
        """
        prompt = self._build_generation_prompt(
            task_type=task_type,
            data_info=data_info,
            blueprint=blueprint,
            iteration=iteration,
            memory_context=memory_context,
        )

        try:
            llm_client = self._get_llm_client()
            response = await llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self._get_system_prompt(task_type)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=128000,
            )

            result = self._parse_code_response(response)

            logger.info(f"[GENERATE] Parsed code: {len(result.code)} chars, {len(result.code.splitlines())} lines, strategy={result.reasoning}")
            self.generation_history.append({
                'iteration': iteration,
                'task_type': task_type,
                'prompt': prompt,
                'result': result,
            })

            logger.info(f"Code generated for {task_type} (iteration {iteration})")
            return result

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    async def fix_code(
        self,
        original_code: str,
        error_message: str,
        task_type: str,
        data_info: Dict[str, Any],
        iteration: int,
        blueprint: Optional[Dict[str, Any]] = None,
        memory_context: str = "",
    ) -> CodeGenerationResult:
        """
        Fix code based on error.

        Args:
            original_code: Original code
            error_message: Error message
            task_type: Task type
            data_info: Data information
            iteration: Current iteration number
            blueprint: Architecture blueprint (optional)
            memory_context: Memory context

        Returns:
            CodeGenerationResult: Fixed code
        """

        # Get task instruction (consistent with generate_code)
        task_instruction = _TASK_INSTRUCTIONS.get(
            task_type, f"## Task: {task_type}\nFix the code based on the error."
        )

        # Blueprint information
        blueprint_info = ""
        if blueprint:
            blueprint_info = f"""
## Architecture Blueprint (Reference)
```json
{json.dumps(blueprint, indent=2)}
```

Use this blueprint as reference for the correct model architecture. If the error is about missing class definitions, refer to the blueprint for the expected class names and structure.
"""

        # Memory information
        memory_info = ""
        if memory_context:
            memory_info = f"""
## Previous Experience
{memory_context}

Learn from previous successes and failures to avoid repeating the same mistakes.
"""

        # Parse error message, extract line number and context
        exact_line_info = ""
        import re
        match = re.search(r'line (\d+)', error_message)
        if match:
            line_num = int(match.group(1))
            lines = original_code.split('\n')
            if 0 < line_num <= len(lines):
                line_content = lines[line_num - 1]
                # Extract 2 lines before and after the error line as indentation context
                ctx_start = max(0, line_num - 3)
                ctx_end = min(len(lines), line_num + 2)
                context_lines = lines[ctx_start:ctx_end]
                context_block = '\n'.join(f"{ctx_start + 1 + i:4d}| {l}" for i, l in enumerate(context_lines))

                # Check if there's an indentation issue
                has_indent_issue = any(kw in error_message.lower() for kw in [
                    'indentation', 'indented', 'indent', 'expected an indented block'
                ])

                indent_warning = ""
                if has_indent_issue:
                    indent_warning = """
**INDENTATION FIX REQUIRED:** This is an indentation error.
- Count EXACTLY how many spaces the previous line uses for indentation
- Use the SAME number of spaces for the new/fixed line
- Never mix tabs and spaces"""

                exact_line_info = f"""
## ERROR CONTEXT (CRITICAL)
The error is at line {line_num}. Surrounding context:
```
{context_block}
```

Line {line_num} (the error line):
```python
{line_content}
```
{indent_warning}
You must fix THIS LINE only. Keep all surrounding lines EXACTLY as they are.
"""

        # Build complete fix prompt (compute fix_history before building prompt)
        error_key = self._get_error_key(error_message)
        fix_history = self._build_fix_history(original_code, error_message)

        # Check if same error fixed 3+ times without success (skip LLM call)
        if self._fix_attempt_count.get(error_key, 0) >= 3:
            logger.warning(f"Same error fixed 3+ times without success: {error_key[:50]}...")
            raise RuntimeError(f"Same error fixed 3+ times without success: {error_key[:100]}")

        try:
            prompt = f"""
{task_instruction}

{blueprint_info}

## Original Code (FAILED)
```python
{original_code}
```

## Error Message
```
{error_message}
```

{memory_info}

{fix_history}
{exact_line_info}

## Fix Strategy
Fix the error based on the error message. Your fixed code MUST end with `result = {...}` containing all metrics. Do NOT stop after partial code.

{_RESPONSE_FORMAT_BLOCK}
"""

            messages=[
                {"role": "system", "content": self._get_system_prompt(task_type)},
                {"role": "user", "content": prompt}
            ]
            response = await self._get_llm_client().chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=128000,
            )

            result = self._parse_code_response(response)

            # Save original LLM response for debugging
            self._save_llm_response_for_debug(response, f"fix_iter{iteration}")

            # Track fix attempt count
            self._fix_attempt_count[error_key] = self._fix_attempt_count.get(error_key, 0) + 1

            self.generation_history.append({
                'iteration': iteration,
                'task_type': task_type,
                'fix_iteration': True,
                'original_code': original_code,
                'error_message': error_message,
                'fixed_code': result.code,
            })

            logger.info(f"Code fixed for {task_type} (iteration {iteration})")
            return result

        except Exception as e:
            logger.error(f"Code fixing failed: {e}")
            return self._simple_fix(original_code, error_message)

    def _build_generation_prompt(
        self,
        task_type: str,
        data_info: Dict[str, Any],
        blueprint: Optional[Dict[str, Any]],
        iteration: int,
        memory_context: str,
    ) -> str:
        """Build code generation prompt"""

        task_instruction = _TASK_INSTRUCTIONS.get(
            task_type, f"## Task: {task_type}\nGenerate appropriate code for this task."
        )

        blueprint_info = ""
        if blueprint:
            # Ensure blueprint is valid JSON format
            if isinstance(blueprint, dict):
                try:
                    blueprint_json = json.dumps(blueprint, indent=2)
                    blueprint_info = f"""
## Architecture Blueprint (JSON format required)
```json
{blueprint_json}
```

Use the blueprint as guidance for your implementation.
"""
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize blueprint to JSON: {e}, using str representation")
                    blueprint_info = f"""
## Architecture Blueprint
```json
{{json.dumps({{"raw_blueprint": str(blueprint)}}, indent=2)}}
```

Use the blueprint as guidance for your implementation.
"""
            else:
                logger.warning(f"Blueprint is not a dict ({type(blueprint)}), attempting to serialize")
                try:
                    blueprint_info = f"""
## Architecture Blueprint
```json
{{json.dumps({{"raw_blueprint": str(blueprint)}}, indent=2)}}
```

Use the blueprint as guidance for your implementation.
"""
                except:
                    blueprint_info = ""

        memory_info = ""
        if memory_context:
            memory_info = f"""
## Previous Experience
{memory_context}

Learn from previous successes and failures.
"""

        data_prompt = ""
        if 'prompt_context' in data_info:
            data_prompt = data_info['prompt_context']
        else:
            data_prompt = f"""
## Data Information
{json.dumps(data_info, indent=2, default=str)[:2000]}
"""

        return f"""{task_instruction}

{blueprint_info}

{data_prompt}

{memory_info}

## METRIC FORMAT (MANDATORY)
{"You MUST compute JOINT metrics (nmi, ari on ALL cells together using KMeans) AND per-modality diagnostics (nmi_rna, ari_rna, nmi_atac, ari_atac). Always include: nmi, ari, nmi_rna, ari_rna, nmi_atac, ari_atac, asw_celltype, asw_batch, knn_cross." if data_info.get("metric_format") == "per_modality" else "Use nmi, ari, asw_celltype, asw_batch, knn_cross."}

## Available Variables (ALWAYS use these exact names)
The following variables are already injected into the execution environment:
- `adata`: AnnData object (if single data)
- `adata_rna`: RNA AnnData object (if multi-omics)
- `adata_atac`: ATAC AnnData object (if multi-omics)
- `np`: NumPy (ALWAYS use `np.XXX`, not `numpy.XXX`)
- `pd`: Pandas (ALWAYS use `pd.XXX`, not `pandas.XXX`)
- `torch`: PyTorch
- `nn`: PyTorch neural network module (assign `nn = torch.nn` FIRST, then use `nn.Linear`, `nn.Module`, etc.)
- `ad`: AnnData
- GPU/CUDA: AVAILABLE - ALWAYS use GPU for PyTorch operations

## Data Dimension Variables (MUST READ - AUTO-PROBED AT RUNTIME)
**The following global variables are automatically set by the system before your code runs.
Your code MUST use these instead of hardcoding dimensions.**

| Variable | Description |
|----------|-------------|
| `_GLOBAL_N_CELLS` | Number of cells (primary dataset) |
| `_GLOBAL_N_GENES` | Number of genes/features (primary dataset) |
| `_GLOBAL_RNA_N_CELLS` | Number of cells in RNA data |
| `_GLOBAL_RNA_N_GENES` | Number of genes in RNA data |
| `_GLOBAL_ATAC_N_CELLS` | Number of cells in ATAC data |
| `_GLOBAL_ATAC_N_PEAKS` | Number of peaks in ATAC data |
| `_GLOBAL_N_CELL_TYPES` | Number of unique cell types |
| `_GLOBAL_N_BATCHES` | Number of unique batch labels |
| `_GLOBAL_N_PERTURBATIONS` | Number of unique perturbation conditions |
| `_GLOBAL_CELL_TYPES` | Array of cell type labels (use with LabelEncoder if needed) |

**CRITICAL WARNING FOR MULTI-OMICS:**
ATAC data can have hundreds of thousands of peaks (e.g., 491K). NEVER do `adata_atac.X.toarray()` or `torch.tensor(adata_atac.X)` — this will cause CUDA OOM.
ALWAYS reduce ATAC dimensionality first (e.g., LSI/PCA to 50 components) BEFORE feeding into any neural network.

**WRONG (will cause CUDA OOM):**
```python
model = nn.Linear(2000, 512)  # hardcoded, may not match data
X_atac = adata_atac.X.toarray()  # 491K peaks × cells → OOM!
model = nn.Linear(491437, 512)  # too many parameters → OOM!
```

**CORRECT:**
```python
n_genes = _GLOBAL_RNA_N_GENES or _GLOBAL_N_GENES
n_peaks = _GLOBAL_ATAC_N_PEAKS
n_cell_types = _GLOBAL_N_CELL_TYPES

# Reduce ATAC dimensionality FIRST
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=50)
atac_reduced = lsi.fit_transform(adata_atac.X)  # stays sparse, no OOM

# Then use reduced dimensions
encoder_rna = nn.Linear(n_genes, 256)
encoder_atac = nn.Linear(50, 256)  # 50 LSI components, NOT 491K peaks
```

## Code Style Requirements (CRITICAL - READ CAREFULLY)

### 1. Imports (MANDATORY)
- ALWAYS import what you use
- For PyTorch: ALWAYS assign `nn = torch.nn` FIRST, then use `nn.Linear`, `nn.Module`, etc.
- For sklearn: use `from sklearn.decomposition import PCA` then `PCA(...)`
- If you define a custom Dataset class (e.g. `class Dataset(torch.utils.data.Dataset)`), you MUST add `from torch.utils.data import Dataset` at the top — the base classes `torch`, `nn`, `np`, `pd`, `ad` are pre-injected but `Dataset`, `DataLoader`, `TensorDataset` are NOT — never assume these are already available

### 2. GPU/CUDA (MANDATORY)
- ALWAYS set `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- ALWAYS move models to GPU: `model = model.to(device)`
- ALWAYS move tensors to GPU: `tensor = tensor.to(device)`
- Use `.to(device)` for ALL tensor operations
- Print device info: `print(f"Using device: {{device}}")`

### 4. TRAINING PROGRESS & LOGGING (CRITICAL)
**Add these print statements so you can see training progress:**

```python
# At the START of training:
print(f"Starting training: {{epochs}} epochs, {{n_samples}} samples")
print(f"Model parameters: {{sum(p.numel() for p in model.parameters())}}")

# Every 10 epochs:
if epoch % 10 == 0 or epoch == epochs - 1:
    print(f"Epoch {{epoch}}/{{epochs}} - Loss: {{loss.item():.4f}}")

# At the END of training:
print(f"Training complete! Best loss: {{best_loss:.4f}}")
```

### 5. ADAPTIVE EPOCHS (CRITICAL - USE ADEQUATE EPOCHS)
**INSUFFICIENT EPOCHS = POOR RESULTS:**

| Model | Dataset | Epochs |
|-------|---------|--------|
| Simple (1-2 layers) | <10K cells | 50-100 |
| Medium (3-5 layers) | Any | 100-200 |
| Deep (6+ layers) | Any | 200-500 |

```python
n_params = sum(p.numel() for p in model.parameters())
n_samples = X.shape[0]
if n_params < 10000:
    epochs = max(100, min(200, n_samples // 100))
elif n_params < 100000:
    epochs = max(200, min(300, n_samples // 50))
else:
    epochs = max(300, min(500, n_samples // 30))
print(f"Training for {{epochs}} epochs")
```

### 6. Early Stopping
DO NOT implement early stopping manually. Let the model train for the full number of epochs. If you must add early stopping, you MUST define and increment a counter:

```python
patience_counter = 0
for epoch in range(epochs):
    # training...
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0  # Reset counter when improving
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping at epoch", epoch)
            break
```

### 7. CLASS DEFINITIONS (ZERO TOLERANCE)
**YOU MUST DEFINE EVERY SINGLE CLASS BEFORE USING IT. NO EXCEPTIONS.**

**REQUIRED CODE STRUCTURE (FOLLOW EXACTLY):**
```python
# STEP 0: ALWAYS assign nn alias FIRST, before any class definition
nn = torch.nn

# STEP 1: Define ALL your classes, RIGHT AFTER the nn alias
class ControlEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x))

class PerturbationEncoder(nn.Module):
    def __init__(self, n_drugs, n_doses, emb_dim):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.dose_emb = nn.Embedding(n_doses, emb_dim)

    def forward(self, drug_ids, dose_ids):
        return self.drug_emb(drug_ids) + self.dose_emb(dose_ids)

# STEP 2: ONLY after ALL classes are defined, create instances and use them
control_encoder = ControlEncoder(input_dim, hidden_dim)
perturbation_encoder = PerturbationEncoder(n_drugs, n_doses, emb_dim)
```

**FORBIDDEN PATTERNS (WILL CAUSE ERRORS):**
- [X] Using a class before defining it
- [X] Assuming any class exists in the environment
- [X] Using `self.encoder = SomeClass(...)` without defining `SomeClass`
- [X] Forward references to undefined classes

**MANDATORY CHECKLIST BEFORE OUTPUTTING CODE:**
1. [ ] Scan through your entire code
2. [ ] Verify `nn = torch.nn` appears at the very top (before any `nn.` usage)
3. [ ] Identify every class instantiation (e.g., `XxxClass(...)`)
4. [ ] Verify that `class XxxClass:` appears BEFORE its first usage
5. [ ] If any class is used but not defined, ADD THE DEFINITION NOW
6. [ ] All classes must have `__init__` and `forward` methods

**COMMON CLASSES YOU MUST DEFINE (if used):**
- `ControlEncoder`, `ExpressionEncoder`, `PerturbationEncoder`
- `DrugEncoder`, `CellTypeEncoder`, `PathwayEncoder`
- `FiLMGenerator`, `MLPDecoder`, `AttentionModule`
- `GeneGraphGAT`, `FiLMDecoder`
- Any other custom class you reference

**CRITICAL DEBUGGING STEPS BEFORE OUTPUT:**
1. Search your code for ALL class instantiations (lines with `SomeClass(`)
2. For each instantiation, verify the class is defined above it
3. Print the first 50 lines of your code to verify structure
4. Ensure `nn = torch.nn` is at the VERY TOP of your code

**VERIFICATION PATTERN:** Add this debug code at the beginning to verify class order:
\\```python
# Debug: verify all classes are defined before use
import inspect
_debug_classes = [name for name, obj in list(globals().items()) if inspect.isclass(obj)]
print(f"Debug check passed")
\\```

### 8. Code Completeness + Truncation Safety (CRITICAL)

**Two hard rules — follow BOTH:**

Rule 1 — The code must be COMPLETE:
- No undefined references, no missing imports, no missing class definitions
- The LAST non-empty line of your code MUST be `result = {{...}}` with all metrics
- DO NOT stop at a class/function definition — you MUST train, predict, and return metrics
- If your code ends without calling the model and computing `result = {{...}}`, it is REJECTED

Rule 2 — Place `result = {{...}}` EARLY as truncation protection:
- After defining classes and imports, immediately write: `result = {{"status": "placeholder"}}`
- This ensures partial code always has a return value even if truncated
- Overwrite it with the real result at the end: `result = {{"status": "success", "nmi": ...}}`
- The LAST line of your output must be the real `result = {{...}}`, not the placeholder

### 9. Debugging
- Print informative messages about progress
- Include shape information for tensors

{_RESPONSE_FORMAT_BLOCK}
"""

    def _get_system_prompt(self, task_type: str) -> str:
        return "Generate Python code for bioinformatics analysis. Output JSON with code field."

    # ===================================================================== #
    # Code Parsing Entry
    # ===================================================================== #

    def _validate_and_return(self, result: CodeGenerationResult, strategy_name: str) -> CodeGenerationResult:
        """
        Perform syntax pre-check before returning parsed result.
        If the generated code has a SyntaxError (e.g., LLM failed in json_plus_code format,
        causing code block content to not be truly fixed), reject this strategy directly,
        letting the parser try other strategies or raise a clear error, instead of letting
        code with SyntaxError enter the execution loop.
        """
        return _validate_and_return_static(result, strategy_name, _DIMENSION_PROBE_CODE)
    
    def _save_code_for_debug(self, code: str, prefix: str):
        """Save code to temporary file for debugging"""
        try:
            from datetime import datetime
            import os
            
            # Use a more general path, compatible with Windows
            if os.path.exists("/root/autodl-tmp"):
                debug_dir = Path("/root/autodl-tmp/BioAgent/debug")
            else:
                debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = debug_dir / f"{prefix}_{timestamp}.py"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.debug(f"Code saved to: {filepath}")
        except Exception as e:
            logger.debug(f"Failed to save debug code: {e}")

    def _save_llm_response_for_debug(self, response: str, prefix: str):
        """Save LLM raw response for debugging"""
        try:
            from datetime import datetime
            import os
            
            if os.path.exists("/root/autodl-tmp"):
                debug_dir = Path("/root/autodl-tmp/BioAgent/debug")
            else:
                debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = debug_dir / f"llm_response_{prefix}_{timestamp}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.debug(f"LLM response saved to: {filepath}")
        except Exception as e:
            logger.debug(f"Failed to save LLM response: {e}")

    def _parse_code_response(self, response: str) -> CodeGenerationResult:
        """
        Parse LLM response and extract code. Error correction is delegated to LLM during execution.

        Parsing strategies (by priority):
        1.  JSON + code block separated format (new recommended format)
        2.  Full JSON parsing
        3.  Truncated JSON extraction
        4.  Markdown python code block
        5.  Markdown generic code block
        6.  Full text scan to extract Python code (last resort)
        """
        self._save_llm_response_for_debug(response, "raw")
        cleaned = self._clean_response(response)

        strategies = [
            (self._try_parse_json_plus_code, "json_plus_code"),
            (self._try_parse_full_json, "full_json"),
            (self._try_parse_truncated_json, "truncated_json"),
            (self._try_extract_markdown_block, "markdown_python", "```python"),
            (self._try_extract_markdown_block, "markdown_generic", "```"),
            (self._try_extract_python_from_raw_text, "raw_text"),
        ]

        for s in strategies:
            if len(s) == 2:
                fn, label = s
                result = fn(cleaned)
            else:
                fn, label, arg = s
                result = fn(cleaned, arg)
            if result:
                logger.info(f"[parse] Success via {label}")
                return _validate_and_return_static(result, label, _DIMENSION_PROBE_CODE)

        logger.warning(f"All parsing strategies failed. Preview: {repr(cleaned[:300])}")
        raise ValueError("All code parsing strategies failed")

    # ===================================================================== #
    # Strategy Methods
    # ===================================================================== #

    def _try_parse_full_json(self, cleaned: str) -> Optional[CodeGenerationResult]:
        """Strategy 1: Full JSON parsing (for legacy code field format compatibility)"""
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and 'code' in data:
                # Legacy format: code inside JSON string (no longer recommended)
                code = self._safe_extract_code(data['code'])
                return CodeGenerationResult(
                    code=code,
                    reasoning=data.get('reasoning', ''),
                    dependencies=data.get('dependencies', []),
                    estimated_complexity=data.get('estimated_complexity', 'medium'),
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _try_parse_truncated_json(self, cleaned: str) -> Optional[CodeGenerationResult]:
        """
        Strategy 1.5: Handle truncated JSON (legacy format, kept as fallback).
        """
        if '"code":' not in cleaned:
            return None

        # Find content after "code":
        idx = cleaned.find('"code":')
        if idx < 0:
            return None

        after = cleaned[idx + 7:].lstrip()
        if not after.startswith('"'):
            return None

        # Parse escaped string character by character
        result = []
        i = 1  # Skip opening quote
        while i < len(after):
            ch = after[i]
            if ch == '\\' and i + 1 < len(after):
                nxt = after[i + 1]
                escape_map = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', "'": "'", 'u003c': '<', 'u003e': '>'}
                result.append(escape_map.get(nxt, nxt))
                i += 2
            elif ch == '"':
                break
            else:
                result.append(ch)
                i += 1

        code = ''.join(result).strip()
        if not code or len(code) < 20 or not self._looks_like_python_code(code):
            return None

        reasoning = self._extract_truncated_field(cleaned, 'reasoning')
        deps_raw = self._extract_truncated_field(cleaned, 'dependencies')
        complexity = self._extract_truncated_field(cleaned, 'estimated_complexity')

        dependencies = []
        if deps_raw:
            try:
                dependencies = json.loads(deps_raw)
                if not isinstance(dependencies, list):
                    dependencies = []
            except (json.JSONDecodeError, TypeError):
                list_match = re.search(r'\[([^\]]*)\]', deps_raw)
                if list_match:
                    items = [i.strip().strip('"\'') for i in list_match.group(1).split(',') if i.strip()]
                    dependencies = items

        if complexity not in ('low', 'medium', 'high'):
            complexity = 'medium'

        return CodeGenerationResult(
            code=code,
            reasoning=f'Truncated JSON: {reasoning or "code field recovered"}',
            dependencies=dependencies,
            estimated_complexity=complexity,
        )

    def _extract_truncated_field(self, text: str, field: str) -> Optional[str]:
        """Extract specified field value (string type) from truncated JSON"""
        for pattern in (
            rf'"{field}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
            rf'"{field}"\s*:\s*([^\s,\}}]+)',
        ):
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
        return None

    def _try_parse_json_plus_code(self, cleaned: str) -> Optional[CodeGenerationResult]:
        """
        Strategy 2.5: JSON + code block separated format.

        New format: JSON first, ```python code block after.
        Example:
        {"reasoning": "...", "dependencies": [], "estimated_complexity": "medium"}
        ```python
        import numpy as np
        ...
        ```
        """
        # Find position of first {
        first_brace = cleaned.find('{')
        if first_brace == -1:
            return None

        # Find JSON end position (find matching })
        depth = 0
        json_end = -1
        in_string = False
        escape_next = False

        for i in range(first_brace, len(cleaned)):
            ch = cleaned[i]

            if escape_next:
                escape_next = False
                continue

            if ch == '\\':
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if json_end == -1:
            return None

        json_str = cleaned[first_brace:json_end].strip()
        after_json = cleaned[json_end:].strip()

        # Parse JSON metadata
        try:
            metadata = json.loads(json_str)
        except json.JSONDecodeError:
            # JSON parsing failed, try to fix
            fixed = self._fix_unescaped_special_chars(json_str)
            try:
                metadata = json.loads(fixed)
            except json.JSONDecodeError:
                return None

        # Find code block
        code = ""
        reasoning = metadata.get('reasoning', '')
        dependencies = metadata.get('dependencies', [])
        complexity = metadata.get('estimated_complexity', 'medium')

        # Find code blocks starting with ```python or ``` (prefer the last one to avoid short import block interference)
        all_blocks = []
        for prefix in ('```python', '```'):
            pos = 0
            while True:
                block_start = after_json.find(prefix, pos)
                if block_start == -1:
                    break
                line_end = after_json.find('\n', block_start)
                block_end = after_json.find('```', line_end + 1 if line_end != -1 else block_start + len(prefix))
                if block_end != -1:
                    block_content = after_json[block_start + len(prefix):block_end].strip()
                    # Filter out language identifier at the start of code block (e.g., "python" after ```python)
                    if block_content and block_content.split('\n')[0].strip() in ('python', 'json', 'yaml', 'txt'):
                        block_content = '\n'.join(block_content.split('\n')[1:])
                    all_blocks.append(block_content)
                    pos = block_end + 3
                else:
                    break

        # Select the longest code block (usually the most important complete implementation)
        if all_blocks:
            code = max(all_blocks, key=len)
        else:
            code = ""

        # If no code block found, try to extract Python code from after_json
        if not code and after_json:
            # Find Python keywords like import/class/def
            lines = after_json.split('\n')
            code_lines = []
            started = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from ') or \
                   stripped.startswith('class ') or stripped.startswith('def ') or \
                   stripped.startswith('nn = ') or stripped.startswith('if ') or \
                   stripped.startswith('# AUTO'):
                    started = True
                if started:
                    code_lines.append(line)

            if code_lines:
                code = '\n'.join(code_lines).strip()

        if code and self._looks_like_python_code(code):
            return CodeGenerationResult(
                code=code,
                reasoning=reasoning or 'Extracted from JSON + code block format',
                dependencies=dependencies if isinstance(dependencies, list) else [],
                estimated_complexity=complexity if complexity in ('low', 'medium', 'high') else 'medium',
            )

        return None

    def _try_extract_markdown_block(
        self, cleaned: str, prefix: str
    ) -> Optional[CodeGenerationResult]:
        """Strategy 4/5: Extract ```xxx ... ``` code blocks, prefer the longest block"""
        block = self._extract_all_code_blocks(cleaned, prefix)
        if not block:
            return None

        # Recursively parse within block
        for strategy_fn, label in (
            (self._try_parse_full_json, 'full JSON'),
            (self._try_parse_json_plus_code, 'JSON string'),
            (self._try_extract_python_from_raw_text, '"code" field'),
        ):
            result = strategy_fn(block)
            if result:
                result.reasoning = f'{prefix} block -> {label}'
                return result

        # Block is pure code
        code = self._safe_extract_code(block)
        if self._looks_like_python_code(code):
            return CodeGenerationResult(
                code=code,
                reasoning=f'Extracted from {prefix} code block',
                dependencies=[],
                estimated_complexity='medium',
            )
        return None

    def _try_extract_python_from_raw_text(self, text: str) -> Optional[CodeGenerationResult]:
        """
        Strategy 7: Full text scan to extract Python code (last resort).

        Scan all lines, find the first contiguous Python code block,
        and select the longest from candidates.
        """
        python_kw = ('import ', 'from ', 'class ', 'def ', 'async def ',
                     'for ', 'while ', 'with ', '@', 'if ')
        natlang = ('the ', 'this ', 'here ', 'sure ', 'okay ', 'i will',
                   'i\'ll', 'let me', 'first ', 'next ', 'finally ',
                   'step ', 'result', 'summary', 'output', 'below ')
        extra_code = ('pass', 'break', 'continue', 'return', 'self.',
                      'else:', 'elif ', '    ', '\t')

        candidates = []
        in_code = False
        code_lines = []

        for line in text.strip().split('\n'):
            stripped = line.strip()
            # Skip markdown fence lines
            if stripped.startswith('```') or stripped.endswith('```'):
                if in_code:
                    candidates.append('\n'.join(code_lines))
                    code_lines = []
                    in_code = False
                continue

            lower = stripped.lower()
            if not in_code and lower and any(lower.startswith(m) for m in natlang):
                if code_lines:
                    candidates.append('\n'.join(code_lines))
                    code_lines = []
                continue

            is_code = any(stripped.startswith(kw) for kw in python_kw)
            if not is_code and stripped and not stripped.startswith('#'):
                if stripped in extra_code[:4] or \
                   any(stripped.startswith(e) for e in extra_code[4:]):
                    is_code = True

            if is_code:
                if not in_code:
                    in_code = True
                    code_lines = []
                code_lines.append(line)
            elif in_code:
                candidates.append('\n'.join(code_lines))
                code_lines = []
                in_code = False

        if code_lines:
            candidates.append('\n'.join(code_lines))

        # Pick the LAST candidate (LLM's main code block is always at the end)
        best = candidates[-1] if candidates else ''
        if best and self._looks_like_python_code(best) and len(best) > 30:
            return CodeGenerationResult(
                code=best,
                reasoning=f'Raw text: best of {len(candidates)} candidates, len={len(best)}',
                dependencies=[],
                estimated_complexity='medium',
            )
        return None

    # ===================================================================== #
    # Utility Methods
    # ===================================================================== #

    def _extract_first_code_block(self, text: str, prefix: str) -> str:
        """Extract content of the first ```xxx ... ``` code block"""
        start = text.find(prefix)
        if start == -1:
            return ""
        line_end = text.find('\n', start)
        if line_end == -1:
            return ""
        end = text.find('```', line_end + 1)
        if end == -1:
            return ""
        return text[line_end + 1:end].strip()

    def _extract_all_code_blocks(self, text: str, prefix: str) -> List[str]:
        """Extract content of all ```xxx ... ``` code blocks, return the longest block"""
        all_blocks = []
        pos = 0
        while True:
            start = text.find(prefix, pos)
            if start == -1:
                break
            line_end = text.find('\n', start)
            end = text.find('```', line_end + 1 if line_end != -1 else start + len(prefix))
            if end == -1:
                break
            block = text[line_end + 1:end].strip()
            lines = block.split('\n')
            if lines and lines[0].strip() in ('python', 'json', 'yaml', 'txt'):
                block = '\n'.join(lines[1:])
            if block:
                all_blocks.append(block)
        if not all_blocks:
            return []
        return max(all_blocks, key=len)

    def _extract_last_code_block(self, text: str) -> str:
        """
        Extract content of the last ```...``` code block in text.
        Search from back to front to avoid interference from natural language code blocks at the beginning.
        """
        last_end = -1
        pos = 0
        while True:
            idx = text.find('```', pos)
            if idx == -1:
                break
            last_end = idx
            pos = idx + 3
        if last_end == -1:
            return ""
        start = text.rfind('```', 0, last_end)
        if start == -1:
            return ""
        line_end = text.find('\n', start)
        if line_end == -1 or line_end > last_end:
            return ""
        return text[line_end + 1:last_end].strip()

    def _extract_string_value(self, text: str, start: int) -> Tuple[str, int]:
        """
        Extract paired string value starting from text[start], supporting escapes.

        Returns: (decoded_string, index_after_closing_quote) or ("", -1)
        """
        i = start
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                raw = text[start:i]
                decoded = (raw
                    .replace('\\n', '\n')
                    .replace('\\t', '\t')
                    .replace('\\"', '"')
                    .replace('\\\\', '\\'))
                return decoded, i + 1
            i += 1
        return "", -1

    def _safe_extract_code(self, raw: str) -> str:
        """Remove XML/agent tags and markdown delimiters"""
        cleaned = raw
        lines = cleaned.split('\n')
        result_lines = [
            line for line in lines
            if line.strip() not in ('```', '```json', '```python', '```yaml', '```txt', 'python', 'json', 'yaml', 'txt')
        ]
        code = '\n'.join(result_lines).strip()
        return code

    def _looks_like_python_code(self, code: str) -> bool:
        """Check if text contains Python code characteristics, excluding import-only code stubs"""
        if not code or len(code) < 50:
            return False
        # Check for basic Python keywords
        python_patterns = [
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+',
            r'\bfor\s+\w+\s+in\s',
            r'\bif\s+.*:',
            r'\b(import|from)\s+\w+',  # Downgraded to weak check
        ]
        found_def_class_for = False
        found_import = False
        for pattern in python_patterns:
            if re.search(pattern, code):
                if pattern.startswith(r'\bdef') or pattern.startswith(r'\bclass') or pattern.startswith(r'\bfor'):
                    found_def_class_for = True
                elif pattern.startswith(r'\b(import'):
                    found_import = True
        if not found_def_class_for:
            return False
        return True

    def _clean_response(self, response: str) -> str:
        cleaned = re.sub(r'</?\w+(:\w+)*>', '', response)
        return cleaned

    def _fix_unescaped_special_chars(self, text: str) -> str:
        """
        Fix unescaped special characters (< and >) in JSON.

        When LLM outputs Python code in JSON "code" string values,
        < and > characters may not be properly escaped, causing JSON parsing failure.

        Strategy: Scan text, find unescaped < and > in JSON string values,
        and replace them with \\u003c and \\u003e.
        """
        import re
        result = []
        i = 0
        in_string = False

        while i < len(text):
            ch = text[i]

            if ch == '"' and (i == 0 or text[i - 1] != '\\'):
                # Encounter unescaped quote, toggle string state
                in_string = not in_string
                result.append(ch)
                i += 1
            elif ch == '\\' and in_string and i + 1 < len(text):
                # Escape sequence (\n, \t, \", \\, \u003c etc.), keep as-is
                result.append(ch)
                result.append(text[i + 1])
                i += 2
            elif ch in '<>' and in_string:
                # Inside string value, unescaped < or >, need to escape
                if ch == '<':
                    result.append('\\u003c')
                else:
                    result.append('\\u003e')
                i += 1
            else:
                result.append(ch)
                i += 1

        return ''.join(result)

    def _get_error_key(self, error_message: str) -> str:
        import re
        error_type = re.search(r'(\w+Error|\w+Exception):', error_message)
        error_type = error_type.group(1) if error_type else "Unknown"
        error_snippet = error_message[:200].replace('\n', ' ').strip()
        return f"{error_type}|{error_snippet}"

    def _build_fix_history(self, original_code: str, error_message: str) -> str:
        """Build fix history information, telling LLM what was tried before"""
        history = []
        error_key = self._get_error_key(error_message)
        attempt_num = 0

        for entry in self.generation_history[-5:]:  # Last 5 attempts
            if entry.get('fix_iteration') and entry.get('error_message'):
                prev_key = self._get_error_key(entry['error_message'])
                if prev_key == error_key:
                    attempt_num += 1
                    fix_type = self._detect_fix_type(entry.get('original_code', ''), entry.get('fixed_code', ''))
                    history.append(f"- Attempt {attempt_num}: {fix_type}")

        if not history:
            return ""

        return f"""
## PREVIOUS FIX ATTEMPTS (DO NOT REPEAT THESE PATTERNS)
{chr(10).join(history[:3])}
"""

    def _detect_fix_type(self, original: str, fixed: str) -> str:
        """Detect fix type"""
        if not original or not fixed:
            return "unknown fix"

        # Detect which parts were modified
        orig_lines = original.split('\n')
        fixed_lines = fixed.split('\n')

        # Simple detection: count line changes
        if len(fixed_lines) > len(orig_lines) * 1.5:
            return "Added many new lines"
        if len(fixed_lines) > len(orig_lines):
            return "Added some new lines"

        # Detect if nn assignment was added
        if 'nn = torch.nn' in fixed and 'nn = torch.nn' not in original:
            return "Added nn alias assignment"

        # Detect if import was added
        orig_imports = set(l.strip() for l in orig_lines if 'import' in l)
        fixed_imports = set(l.strip() for l in fixed_lines if 'import' in l)
        if fixed_imports - orig_imports:
            return f"Added imports: {list(fixed_imports - orig_imports)[0][:30]}"

        return "Modified existing code"

    # ===================================================================== #
    # Fallback
    # ===================================================================== #

    def _generate_fallback_code(self, task_type: str, error: str) -> CodeGenerationResult:
        """When all parsing strategies fail, return a minimal executable code"""
        fallback = {
            "integration": '''
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

result = {"status": "placeholder"}

# Detect paired/unpaired
def check_paired(adata_rna, adata_atac):
    if adata_rna.n_obs != adata_atac.n_obs:
        return False
    try:
        rna_bc = set(adata_rna.obs_names.astype(str))
        atac_bc = set(adata_atac.obs_names.astype(str))
        overlap = rna_bc & atac_bc
        if len(overlap) > 0.5 * min(len(rna_bc), len(atac_bc)):
            return True
    except Exception:
        pass
    return adata_rna.n_obs == adata_atac.n_obs

# Preprocess RNA: normalize_total -> log1p -> PCA(50)
def preprocess_rna(adata, n_pca=50):
    X = adata.X
    if sp.issparse(X):
        lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        lib[lib == 0] = 1.0
        from scipy.sparse import diags
        X = diags(1e4 / lib) @ X
        X.data = np.log1p(X.data)
    else:
        X = np.asarray(X, dtype=np.float32)
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        X = np.log1p(X / lib * 1e4)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X if not sp.issparse(X) else X.toarray())
    return StandardScaler().fit_transform(X_pca).astype(np.float32)

# Preprocess ATAC: TF-IDF -> LSI(50)
def preprocess_atac(adata, n_lsi=50):
    X = adata.X
    n_cells = X.shape[0]
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    cell_depth = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    cell_depth[cell_depth == 0] = 1.0
    tf = X.multiply(1.0 / cell_depth[:, np.newaxis])
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1 + n_cells / (1 + df))
    X_tfidf = tf.multiply(idf)
    lsi = TruncatedSVD(n_components=n_lsi, random_state=42)
    X_lsi = lsi.fit_transform(X_tfidf)
    return StandardScaler().fit_transform(X_lsi).astype(np.float32)

if 'adata_rna' not in globals() or 'adata_atac' not in globals():
    result = {"status": "error", "message": "Both adata_rna and adata_atac required"}
else:
    is_paired = check_paired(adata_rna, adata_atac)
    print(f"Data pairing detected: {'PAIRED' if is_paired else 'UNPAIRED'}")
    strategy = "A (CrossModal+Contrastive)" if is_paired else "B (DANN+Contrastive)"
    print(f"Using Strategy {strategy}")

    X_rna = preprocess_rna(adata_rna)
    X_atac = preprocess_atac(adata_atac)
    n_rna, n_atac = X_rna.shape[0], X_atac.shape[0]
    print(f"RNA: {n_rna} cells, ATAC: {n_atac} cells")

    rna_ct = adata_rna.obs.get('cell_type', adata_rna.obs.get('celltype', None))
    atac_ct = adata_atac.obs.get('cell_type', adata_atac.obs.get('celltype', None))
    if rna_ct is not None and atac_ct is not None:
        rna_labels_str = rna_ct.astype(str).values
        atac_labels_str = atac_ct.astype(str).values
        le = LabelEncoder()
        le.fit(np.concatenate([rna_labels_str, atac_labels_str]))
        rna_labels = le.transform(rna_labels_str)
        atac_labels = le.transform(atac_labels_str)
        n_types = len(le.classes_)
    else:
        rna_labels = np.zeros(n_rna, dtype=int)
        atac_labels = np.zeros(n_atac, dtype=int)
        n_types = 1

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.1)])
                prev = h
            self.encoder = nn.Sequential(*layers)
            self.fc_mu = nn.Linear(prev, latent_dim)
            self.fc_logvar = nn.Linear(prev, latent_dim)
        def forward(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = torch.clamp(self.fc_logvar(h), -10, 10)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return z, mu, logvar

    class CellTypeClassifier(nn.Module):
        def __init__(self, latent_dim, n_types):
            super().__init__()
            self.clf = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, n_types))
        def forward(self, z):
            return self.clf(z)

    class CrossModalDecoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim=128):
            super().__init__()
            self.pred = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        def forward(self, z):
            return self.pred(z)

    class GradientReversalLayer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None

    class ModalityDiscriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dims=[256, 128]):
            super().__init__()
            layers = []
            prev = latent_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(0.3)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, z, alpha=1.0):
            z_rev = GradientReversalLayer.apply(z, alpha)
            return self.net(z_rev)

    latent_dim = 128
    rna_enc = Encoder(X_rna.shape[1], [512, 256], latent_dim).to(device)
    atac_enc = Encoder(X_atac.shape[1], [512, 256, 128], latent_dim).to(device)
    ct_clf = CellTypeClassifier(latent_dim, n_types).to(device)

    if is_paired:
        cross_dec = CrossModalDecoder(latent_dim).to(device)
        all_params = list(rna_enc.parameters()) + list(atac_enc.parameters()) + list(ct_clf.parameters()) + list(cross_dec.parameters())
    else:
        discriminator = ModalityDiscriminator(latent_dim).to(device)
        all_params = list(rna_enc.parameters()) + list(atac_enc.parameters()) + list(ct_clf.parameters()) + list(discriminator.parameters())

    X_rna_t = torch.tensor(X_rna, device=device)
    X_atac_t = torch.tensor(X_atac, device=device)
    rna_labels_t = torch.tensor(rna_labels, device=device)
    atac_labels_t = torch.tensor(atac_labels, device=device)

    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-5)

    batch_size = 512
    epochs = 500
    warmup_epochs = 50
    patience = 50
    best_loss = float('inf')
    patience_counter = 0

    mmd_bw = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] if not is_paired else [1.0, 2.0, 5.0, 10.0, 20.0]
    def mmd_loss_fn(x, y):
        loss = 0.0
        for b in mmd_bw:
            xx = torch.mm(x, x.t()); yy = torch.mm(y, y.t()); xy = torch.mm(x, y.t())
            rx = xx.diag().unsqueeze(0).expand_as(xx)
            ry = yy.diag().unsqueeze(0).expand_as(yy)
            kxx = torch.exp(-b * (rx.t() + rx - 2*xx))
            kyy = torch.exp(-b * (ry.t() + ry - 2*yy))
            kxy = torch.exp(-b * (rx.t() + ry - 2*xy))
            loss = loss + kxx.sum()/(x.shape[0]*x.shape[0]) + kyy.sum()/(y.shape[0]*y.shape[0]) - 2*kxy.sum()/(x.shape[0]*y.shape[0])
        return loss / len(mmd_bw)

    print(f"Training {epochs} epochs, batch_size={batch_size}, latent_dim={latent_dim}")
    for epoch in range(epochs):
        rna_enc.train(); atac_enc.train(); ct_clf.train()
        if is_paired:
            cross_dec.train()
        else:
            discriminator.train()

        kl_weight = min(0.005, epoch / warmup_epochs * 0.005)
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * lr_scale
        elif epoch > warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

        rna_perm = torch.randperm(n_rna, device=device)
        atac_perm = torch.randperm(n_atac, device=device)
        n_batches = max(n_rna, n_atac) // batch_size + 1
        epoch_loss = 0.0

        for b in range(n_batches):
            r_idx = rna_perm[(b*batch_size) % n_rna : (b*batch_size) % n_rna + batch_size]
            a_idx = atac_perm[(b*batch_size) % n_atac : (b*batch_size) % n_atac + batch_size]

            z_r, mu_r, lv_r = rna_enc(X_rna_t[r_idx])
            z_a, mu_a, lv_a = atac_enc(X_atac_t[a_idx])

            kl = -0.5 * torch.mean(1 + lv_r - mu_r.pow(2) - lv_r.exp()) + -0.5 * torch.mean(1 + lv_a - mu_a.pow(2) - lv_a.exp())
            mmd_l = mmd_loss_fn(F.normalize(mu_r, dim=1), F.normalize(mu_a, dim=1))

            temp = 0.3 if is_paired else 0.5
            z_rn = F.normalize(mu_r, dim=1)
            z_an = F.normalize(mu_a, dim=1)
            sim = torch.mm(z_rn, z_an.t()) / temp
            rlab = rna_labels_t[r_idx]; alab = atac_labels_t[a_idx]
            max_cls = max(rlab.max(), alab.max()) + 1
            pos_mask = torch.mm(F.one_hot(rlab, max_cls).float(), F.one_hot(alab, max_cls).float().t())
            n_pos = pos_mask.sum(dim=1).clamp(min=1)
            log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
            cl_loss = -(pos_mask * log_prob).sum(dim=1).mean() / n_pos.mean() if n_types > 1 else torch.tensor(0.0, device=device)

            ct_logits_r = ct_clf(mu_r)
            ct_logits_a = ct_clf(mu_a)
            ce_loss = F.cross_entropy(ct_logits_r, rlab) + F.cross_entropy(ct_logits_a, alab) if n_types > 1 else torch.tensor(0.0, device=device)

            if is_paired:
                pred_a_from_r = cross_dec(mu_r)
                pred_r_from_a = cross_dec(mu_a)
                with torch.no_grad():
                    sim_mat = torch.mm(F.normalize(mu_r, dim=1), F.normalize(mu_a, dim=1).t())
                    nn_map_r2a = sim_mat.argmax(dim=1)
                    nn_map_a2r = sim_mat.argmax(dim=0)
                cross_loss = F.mse_loss(pred_a_from_r, mu_a[nn_map_r2a].detach()) + F.mse_loss(pred_r_from_a, mu_r[nn_map_a2r].detach())
                loss = kl_weight * kl + 0.5 * mmd_l + 2.5 * cl_loss + 0.5 * ce_loss + 0.3 * cross_loss
            else:
                dann_alpha = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
                z_all = torch.cat([mu_r, mu_a], dim=0)
                mod_labels = torch.cat([torch.zeros(mu_r.size(0)), torch.ones(mu_a.size(0))]).to(device)
                d_pred = discriminator(z_all, alpha=dann_alpha)
                dann_loss = F.binary_cross_entropy_with_logits(d_pred.squeeze(), mod_labels)
                loss = kl_weight * kl + 1.0 * mmd_l + 3.0 * cl_loss + 1.0 * ce_loss + 1.0 * dann_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - KL w: {kl_weight:.6f}")

    rna_enc.eval(); atac_enc.eval()
    with torch.no_grad():
        latent_rna = rna_enc(X_rna_t)[1].cpu().numpy()
        latent_atac = atac_enc(X_atac_t)[1].cpu().numpy()

    latent_all = np.concatenate([latent_rna, latent_atac], axis=0)
    ct_all = np.concatenate([rna_labels, atac_labels])
    mod_all = np.concatenate([np.zeros(n_rna), np.ones(n_atac)])

    n_clusters = max(2, n_types)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    pred = km.fit_predict(latent_all)
    nmi = normalized_mutual_info_score(ct_all, pred)
    ari = adjusted_rand_score(ct_all, pred)
    ss = min(5000, len(ct_all))
    asw_ct = silhouette_score(latent_all, ct_all, sample_size=ss) if n_types > 1 else 0.0
    asw_ba = 1 - abs(silhouette_score(latent_all, mod_all, sample_size=ss))

    nn_mod = NearestNeighbors(n_neighbors=min(50, len(latent_all)-1)).fit(latent_all)
    _, nn_idx = nn_mod.kneighbors(latent_all)
    knn_cross = np.mean([np.mean(mod_all[nn_idx[i]] != mod_all[i]) for i in range(len(mod_all))])

    adata_rna.obsm['X_integration'] = latent_rna
    adata_atac.obsm['X_integration'] = latent_atac

    print(f"NMI={nmi:.4f}, ARI={ari:.4f}, ASW_ct={asw_ct:.4f}, ASW_ba={asw_ba:.4f}, KNN={knn_cross:.4f}")

    result = {
        "status": "success",
        "latent_representation": latent_all,
        "predicted_labels": pred,
        "metrics": {
            "nmi": float(nmi),
            "ari": float(ari),
            "nmi_rna": float(normalized_mutual_info_score(rna_labels, pred[:n_rna])),
            "ari_rna": float(adjusted_rand_score(rna_labels, pred[:n_rna])),
            "nmi_atac": float(normalized_mutual_info_score(atac_labels, pred[n_rna:])),
            "ari_atac": float(adjusted_rand_score(atac_labels, pred[n_rna:])),
            "asw_celltype": float(asw_ct),
            "asw_batch": float(asw_ba),
            "knn_cross": float(knn_cross),
        }
    }
''',
            "grn": '''
import numpy as np
from scipy.stats import pearsonr

print("Running fallback GRN code...")

if 'adata' in globals():
    X = adata.X[:100, :50].toarray() if hasattr(adata.X, 'toarray') else adata.X[:100, :50]
    X = np.array(X, dtype=np.float32)
    n_genes = X.shape[1]
    adj = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if np.std(X[:, i]) > 0 and np.std(X[:, j]) > 0:
                corr, _ = pearsonr(X[:, i], X[:, j])
                adj[i, j] = abs(corr)
                adj[j, i] = abs(corr)
    result = {"status": "success", "adj_matrix": adj, "method": "fallback_correlation"}
else:
    result = {"status": "error", "message": "No data available"}
''',
            "perturbation": '''
import numpy as np
import torch
import torch.nn as nn

print("Running fallback perturbation code...")

if 'adata' in globals():
    X = adata.X[:100].toarray() if hasattr(adata.X, 'toarray') else adata.X[:100]
    X = np.array(X, dtype=np.float32)
    model = nn.Linear(X.shape[1], X.shape[1])
    X_tensor = torch.tensor(X)
    with torch.no_grad():
        pred = model(X_tensor)
    result = {"status": "success", "predictions": pred.numpy(), "method": "fallback_linear"}
else:
    result = {"status": "error", "message": "No data available"}
''',
            "genetic_perturbation": '''
import numpy as np
import torch
import torch.nn as nn

print("Running fallback genetic perturbation code...")

if 'adata' in globals():
    X = adata.X[:100].toarray() if hasattr(adata.X, 'toarray') else adata.X[:100]
    X = np.array(X, dtype=np.float32)
    model = nn.Linear(X.shape[1], X.shape[1])
    X_tensor = torch.tensor(X)
    with torch.no_grad():
        pred = model(X_tensor)
    result = {"status": "success", "predictions": pred.numpy(), "method": "fallback_linear"}
else:
    result = {"status": "error", "message": "No data available"}
''',
        }

        code = fallback.get(task_type, f'''
print("Fallback code for task")
result = {{"status": "error", "message": "fallback"}}
''')

        return CodeGenerationResult(
            code=code,
            reasoning=f"Fallback due to: {error}",
            dependencies=[],
            estimated_complexity="low",
        )

    def _simple_fix(self, original_code: str, error_message: str) -> CodeGenerationResult:
        """
        Simple fix for API call exceptions: wrap original code with try-except.
        """
        # NOTE: Do NOT dedent the original code — it may already be top-level
        # and dedenting would remove any existing indentation, breaking the try block.
        # Instead, wrap as-is so that the outer indentation context is preserved.
        if "No module named" in error_message or "is not defined" in error_message:
            fixed_code = f"""
try:
{original_code}
except (ImportError, NameError) as e:
    print(f"Error: {{e}}")
    result = {{"status": "error", "message": str(e)}}
"""
        else:
            fixed_code = f"""
try:
{original_code}
except Exception as e:
    print(f"Error: {{e}}")
    result = {{"status": "error", "message": str(e)}}
"""

        return CodeGenerationResult(
            code=fixed_code,
            reasoning=f"Simple fix: {error_message[:100]}",
            dependencies=[],
            estimated_complexity="low",
        )
