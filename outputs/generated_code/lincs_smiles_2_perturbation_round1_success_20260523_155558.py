"""
Task: perturbation
Round: 1
Generated: 2026-05-23T15:55:58.048389
Iterations: 2
Execution time: 33399.43s
"""


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
# Approach: Conditional Transformer for perturbation response prediction with change-aware loss
nn = torch.nn
import torch
import numpy as np
import pandas as pd
import anndata as ad
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

result = {"status": "placeholder"}

# ====================== CLASS DEFINITIONS ======================
class PerturbationTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, n_perts, n_doses, n_cell_types, n_times, latent_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2, hidden_dim))
        
        self.pert_emb = nn.Embedding(max(n_perts, 1), hidden_dim)
        self.dose_emb = nn.Embedding(max(n_doses, 1), hidden_dim // 2)
        self.cell_type_emb = nn.Embedding(max(n_cell_types, 1), hidden_dim // 2)
        self.time_emb = nn.Embedding(max(n_times, 1), hidden_dim // 4)
        
        self.dose_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        self.cell_type_proj = nn.Linear(hidden_dim // 2, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim // 4, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        
        self.control_proj = nn.Linear(input_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
    
    def forward(self, control_expr, pert_id, dose_id, cell_type_id, time_id):
        control_proj = self.control_proj(control_expr)
        
        pert_embed = self.pert_emb(pert_id)
        dose_embed = self.dose_proj(self.dose_emb(dose_id))
        cell_type_embed = self.cell_type_proj(self.cell_type_emb(cell_type_id))
        time_embed = self.time_proj(self.time_emb(time_id))
        
        condition = pert_embed + dose_embed + cell_type_embed + time_embed
        
        x = torch.stack([control_proj, condition], dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        x = self.transformer(x)
        
        gamma = self.film_gamma(x[:, 0, :])
        beta = self.film_beta(x[:, 0, :])
        
        conditioned = gamma * x[:, 1, :] + beta
        
        output = self.decoder(conditioned)
        
        return output


class PerturbationDataset(Dataset):
    def __init__(self, X, control_mask, pert_encoded, dose_encoded, cell_type_encoded, time_encoded, paired_ctrl_idx, indices):
        self.X = X
        self.control_mask = control_mask
        self.pert_encoded = pert_encoded
        self.dose_encoded = dose_encoded
        self.cell_type_encoded = cell_type_encoded
        self.time_encoded = time_encoded
        self.paired_ctrl_idx = paired_ctrl_idx
        self.indices = indices
        self.control_indices = np.where(control_mask)[0]
        
        # Build mapping from perturbed cell to its paired control
        self.pert_to_ctrl = {}
        for idx in indices:
            if not control_mask[idx]:
                ctrl_idx = paired_ctrl_idx[idx]
                if ctrl_idx < len(control_mask):
                    self.pert_to_ctrl[idx] = ctrl_idx
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        cell_idx = self.indices[idx]
        
        if self.control_mask[cell_idx]:
            control_expr = torch.tensor(self.X[cell_idx], dtype=torch.float32)
            target_expr = control_expr.clone()
        else:
            ctrl_idx = self.pert_to_ctrl.get(cell_idx, cell_idx)
            control_expr = torch.tensor(self.X[ctrl_idx], dtype=torch.float32)
            target_expr = torch.tensor(self.X[cell_idx], dtype=torch.float32)
        
        pert_id = torch.tensor(int(self.pert_encoded[cell_idx]), dtype=torch.long)
        dose_id = torch.tensor(int(self.dose_encoded[cell_idx]), dtype=torch.long)
        cell_type_id = torch.tensor(int(self.cell_type_encoded[cell_idx]), dtype=torch.long)
        time_id = torch.tensor(int(self.time_encoded[cell_idx]), dtype=torch.long)
        # FIX: convert numpy.bool to int to avoid collation error
        is_control = torch.tensor(int(self.control_mask[cell_idx]), dtype=torch.long)
        
        return {
            'control_expr': control_expr,
            'target_expr': target_expr,
            'pert_id': pert_id,
            'dose_id': dose_id,
            'cell_type_id': cell_type_id,
            'time_id': time_id,
            'cell_idx': torch.tensor(cell_idx, dtype=torch.long),
            'is_control': is_control
        }


# ====================== HELPER FUNCTIONS ======================
def detect_perturbation_column(adata):
    candidates = ['perturbation', 'condition', 'treatment', 'drug', 'compound', 'pert_iname_x']
    for col in candidates:
        if col in adata.obs.columns:
            n_unique = len(adata.obs[col].unique())
            if 2 <= n_unique <= 200:
                return col
    cat_cols = adata.obs.select_dtypes(include=['category']).columns
    if len(cat_cols) > 0:
        return cat_cols[0]
    return 'condition'

def detect_cell_type_column(adata):
    candidates = ['cell_type', 'celltype', 'cluster', 'label', 'cell_id']
    for col in candidates:
        if col in adata.obs.columns:
            return col
    return 'cell_type'

def preprocess_data(adata):
    print("Preprocessing data...")
    
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    if 'control' in adata.obs.columns:
        control_mask = adata.obs['control'].values.astype(bool)
    else:
        pert_col = detect_perturbation_column(adata)
        control_labels = ['control', 'ctrl', 'vehicle', 'dmso', 'untreated', 'baseline']
        control_mask = np.array([str(val).lower() in control_labels for val in adata.obs[pert_col]])
    
    pert_col = detect_perturbation_column(adata)
    
    le_pert = LabelEncoder()
    pert_encoded = le_pert.fit_transform(adata.obs[pert_col].astype(str).values)
    n_perts = len(le_pert.classes_)
    
    if 'dose' in adata.obs.columns:
        dose_values = adata.obs['dose'].values.astype(float)
    elif 'dose_val' in adata.obs.columns:
        dose_values = adata.obs['dose_val'].values.astype(float)
    else:
        dose_values = np.ones(len(adata))
    
    n_dose_bins = min(20, len(np.unique(dose_values)))
    dose_bins = np.digitize(dose_values, bins=np.linspace(np.min(dose_values), np.max(dose_values), n_dose_bins))
    le_dose = LabelEncoder()
    dose_encoded = le_dose.fit_transform(dose_bins)
    n_doses = len(le_dose.classes_)
    
    cell_type_col = detect_cell_type_column(adata)
    le_cell = LabelEncoder()
    cell_type_encoded = le_cell.fit_transform(adata.obs[cell_type_col].astype(str).values)
    n_cell_types = len(le_cell.classes_)
    
    if 'pert_time' in adata.obs.columns:
        time_values = adata.obs['pert_time'].values.astype(int)
    else:
        time_values = np.zeros(len(adata), dtype=int)
    le_time = LabelEncoder()
    time_encoded = le_time.fit_transform(time_values)
    n_times = len(le_time.classes_)
    
    if 'paired_control_index' in adata.obs.columns:
        paired_ctrl_idx = []
        for idx, val in enumerate(adata.obs['paired_control_index']):
            try:
                ctrl_idx = np.where(adata.obs.index == val)[0][0]
                paired_ctrl_idx.append(ctrl_idx)
            except:
                paired_ctrl_idx.append(idx)
        paired_ctrl_idx = np.array(paired_ctrl_idx)
    else:
        paired_ctrl_idx = np.arange(len(adata))
    
    if 'split' in adata.obs.columns:
        split_values = adata.obs['split'].values
        train_mask = split_values == 'train'
        test_mask = split_values == 'test'
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
    elif 'random_split' in adata.obs.columns:
        split_values = adata.obs['random_split'].values
        train_mask = split_values == 'train'
        test_mask = (split_values == 'test') | (split_values == 'ood')
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
    else:
        n_cells = len(adata)
        indices = np.random.permutation(n_cells)
        train_size = int(0.8 * n_cells)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    
    print(f"Data size: {X.shape[0]} cells, {X.shape[1]} genes")
    print(f"Control cells: {np.sum(control_mask)}")
    print(f"Perturbations: {n_perts}")
    print(f"Cell types: {n_cell_types}")
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    return {
        'X': X, 'control_mask': control_mask, 'pert_encoded': pert_encoded,
        'dose_encoded': dose_encoded, 'cell_type_encoded': cell_type_encoded,
        'time_encoded': time_encoded, 'paired_ctrl_idx': paired_ctrl_idx,
        'train_indices': train_indices, 'test_indices': test_indices,
        'le_pert': le_pert, 'le_cell': le_cell, 'le_dose': le_dose,
        'le_time': le_time, 'n_perts': n_perts, 'n_doses': n_doses,
        'n_cell_types': n_cell_types, 'n_times': n_times,
        'pert_col': pert_col, 'cell_type_col': cell_type_col
    }


def compute_metrics(y_true, y_pred, de_mask=None):
    metrics = {}
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    metrics['r2'] = float(r2_score(y_true_flat, y_pred_flat))
    metrics['mse'] = float(mean_squared_error(y_true_flat, y_pred_flat))
    pcc, _ = pearsonr(y_true_flat, y_pred_flat)
    metrics['pcc'] = float(pcc)
    
    if de_mask is not None and np.any(de_mask):
        y_true_de = y_true[:, de_mask]
        y_pred_de = y_pred[:, de_mask]
        y_true_de_flat = y_true_de.ravel()
        y_pred_de_flat = y_pred_de.ravel()
        metrics['r2_de'] = float(r2_score(y_true_de_flat, y_pred_de_flat))
        metrics['mse_de'] = float(mean_squared_error(y_true_de_flat, y_pred_de_flat))
        pcc_de, _ = pearsonr(y_true_de_flat, y_pred_de_flat)
        metrics['pcc_de'] = float(pcc_de)
    else:
        metrics['r2_de'] = None
        metrics['mse_de'] = None
        metrics['pcc_de'] = None
    return metrics


def find_top_de_genes(control_expr, perturbed_expr, top_n=20):
    n_genes = control_expr.shape[1]
    t_stats = np.zeros(n_genes)
    for gene_idx in range(n_genes):
        ctrl_vals = control_expr[:, gene_idx]
        pert_vals = perturbed_expr[:, gene_idx]
        ctrl_nonzero = ctrl_vals[ctrl_vals > 0]
        pert_nonzero = pert_vals[pert_vals > 0]
        if len(ctrl_nonzero) >= 5 and len(pert_nonzero) >= 5:
            try:
                t_stat, _ = ttest_ind(ctrl_nonzero, pert_nonzero, equal_var=False)
                t_stats[gene_idx] = abs(t_stat)
            except:
                t_stats[gene_idx] = 0
        else:
            t_stats[gene_idx] = 0
    top_genes = np.argsort(t_stats)[-top_n:]
    return top_genes, t_stats


def pearson_loss(pred, target):
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    cos_sim = nn.functional.cosine_similarity(pred_centered, target_centered, dim=0)
    return 1.0 - cos_sim.mean()


# ====================== MAIN EXECUTION ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_info = preprocess_data(adata)
X = data_info['X']
control_mask = data_info['control_mask']
pert_encoded = data_info['pert_encoded']
dose_encoded = data_info['dose_encoded']
cell_type_encoded = data_info['cell_type_encoded']
time_encoded = data_info['time_encoded']
paired_ctrl_idx = data_info['paired_ctrl_idx']
train_indices = data_info['train_indices']
test_indices = data_info['test_indices']
n_perts = data_info['n_perts']
n_doses = data_info['n_doses']
n_cell_types = data_info['n_cell_types']
n_times = data_info['n_times']

# Normalize expression
print("Normalizing expression...")
X_normalized = np.zeros_like(X, dtype=np.float32)
for i in range(X.shape[0]):
    cell_expr = X[i, :]
    lib_sum = np.sum(cell_expr)
    if lib_sum > 0:
        X_normalized[i, :] = np.log1p(cell_expr / lib_sum * 10000)
    else:
        X_normalized[i, :] = 0

# Create datasets
train_dataset = PerturbationDataset(
    X_normalized, control_mask, pert_encoded, dose_encoded,
    cell_type_encoded, time_encoded, paired_ctrl_idx, train_indices
)
test_dataset = PerturbationDataset(
    X_normalized, control_mask, pert_encoded, dose_encoded,
    cell_type_encoded, time_encoded, paired_ctrl_idx, test_indices
)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

input_dim = X_normalized.shape[1]
hidden_dim = 256
n_heads = 8
n_layers = 4
latent_dim = 64

print(f"Building model with input_dim={input_dim}, hidden_dim={hidden_dim}")
model = PerturbationTransformer(
    input_dim=input_dim, hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
    n_perts=n_perts, n_doses=n_doses, n_cell_types=n_cell_types, n_times=n_times,
    latent_dim=latent_dim
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Find DE genes on training set for loss weighting
print("Finding DE genes for loss weighting...")
train_ctrl_idx = train_indices[control_mask[train_indices]]
train_pert_idx = train_indices[~control_mask[train_indices]]
if len(train_ctrl_idx) > 0 and len(train_pert_idx) > 0:
    control_expr_train = X_normalized[train_ctrl_idx]
    perturbed_expr_train = X_normalized[train_pert_idx]
    de_genes_train, _ = find_top_de_genes(control_expr_train, perturbed_expr_train, top_n=min(50, input_dim))
    de_mask = np.zeros(input_dim, dtype=bool)
    de_mask[de_genes_train] = True
else:
    de_mask = np.zeros(input_dim, dtype=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

print("Starting training...")
epochs = 200
best_loss = float('inf')
patience_counter = 0
patience = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in train_loader:
        control_expr = batch['control_expr'].to(device)
        target_expr = batch['target_expr'].to(device)
        pert_id = batch['pert_id'].to(device)
        dose_id = batch['dose_id'].to(device)
        cell_type_id = batch['cell_type_id'].to(device)
        time_id = batch['time_id'].to(device)
        
        pred_expr = model(control_expr, pert_id, dose_id, cell_type_id, time_id)
        
        loss_mse = nn.MSELoss()(pred_expr, target_expr)
        loss_corr = pearson_loss(pred_expr, target_expr)
        
        if np.any(de_mask):
            loss_de = nn.MSELoss()(pred_expr[:, de_mask], target_expr[:, de_mask])
        else:
            loss_de = loss_mse * 0.0
        
        # Change-aware delta loss
        delta_true = target_expr - control_expr
        delta_pred = pred_expr - control_expr
        loss_delta = nn.MSELoss()(delta_pred, delta_true)
        
        loss = 1.0 * loss_mse + 0.5 * loss_corr + 0.5 * loss_de + 0.3 * loss_delta
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    
    avg_loss = total_loss / max(n_batches, 1)
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"Training complete! Best loss: {best_loss:.4f}")

model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

# Generate predictions
all_pred = []
all_true = []
all_control = []
all_pert_labels = []
all_foldchange_pred = []
all_foldchange_true = []
all_is_control = []

with torch.no_grad():
    for batch in test_loader:
        control_expr = batch['control_expr'].to(device)
        target_expr = batch['target_expr'].to(device)
        pert_id = batch['pert_id'].to(device)
        dose_id = batch['dose_id'].to(device)
        cell_type_id = batch['cell_type_id'].to(device)
        time_id = batch['time_id'].to(device)
        is_control = batch['is_control'].numpy()
        
        pred_expr = model(control_expr, pert_id, dose_id, cell_type_id, time_id)
        
        pred_np = pred_expr.cpu().numpy()
        true_np = target_expr.cpu().numpy()
        ctrl_np = control_expr.cpu().numpy()
        pert_np = pert_id.cpu().numpy()
        
        all_pred.append(pred_np)
        all_true.append(true_np)
        all_control.append(ctrl_np)
        all_pert_labels.append(pert_np)
        all_is_control.append(is_control)
        
        ctrl_safe = np.where(ctrl_np > 0, ctrl_np, 1e-6)
        fc_pred = pred_np / ctrl_safe
        fc_true = true_np / ctrl_safe
        all_foldchange_pred.append(fc_pred)
        all_foldchange_true.append(fc_true)

predicted_expression = np.concatenate(all_pred, axis=0)
true_expression = np.concatenate(all_true, axis=0)
control_expression = np.concatenate(all_control, axis=0)
perturbation_labels = np.concatenate(all_pert_labels, axis=0)
foldchange_predicted = np.concatenate(all_foldchange_pred, axis=0)
foldchange_true = np.concatenate(all_foldchange_true, axis=0)
is_control_arr = np.concatenate(all_is_control, axis=0)

# Find DE genes on test set
print("Finding DE genes for evaluation...")
control_mask_test = control_mask[test_indices]
control_expr_test = true_expression[is_control_arr.astype(bool)]
perturbed_expr_test = true_expression[~is_control_arr.astype(bool)]

if len(control_expr_test) > 0 and len(perturbed_expr_test) > 0:
    de_genes, de_stats = find_top_de_genes(control_expr_test, perturbed_expr_test, top_n=20)
else:
    de_genes = np.arange(20)
    de_stats = np.zeros(20)

de_mask_eval = np.zeros(input_dim, dtype=bool)
de_mask_eval[de_genes] = True

print("Computing metrics...")
# Overall metrics on perturbed cells only
perturbed_mask = ~is_control_arr.astype(bool)
if np.any(perturbed_mask):
    metrics = compute_metrics(true_expression[perturbed_mask], predicted_expression[perturbed_mask], de_mask_eval)
else:
    metrics = compute_metrics(true_expression, predicted_expression, de_mask_eval)

de_gene_names = [adata.var.index[i] for i in de_genes] if hasattr(adata, 'var') and adata.var is not None else [f"Gene_{i}" for i in de_genes]

print(f"Final metrics:")
print(f"R2: {metrics['r2']:.4f}")
print(f"MSE: {metrics['mse']:.4f}")
print(f"PCC: {metrics['pcc']:.4f}")
if metrics['r2_de'] is not None:
    print(f"R2_DE: {metrics['r2_de']:.4f}")
    print(f"MSE_DE: {metrics['mse_de']:.4f}")
    print(f"PCC_DE: {metrics['pcc_de']:.4f}")

result = {
    "status": "success",
    "predicted_expression": predicted_expression,
    "true_expression": true_expression,
    "control_mask": control_mask_test,
    "perturbation_labels": perturbation_labels,
    "de_genes": de_genes.tolist(),
    "de_gene_names": de_gene_names,
    "foldchange_predicted": foldchange_predicted,
    "foldchange_true": foldchange_true,
    "metrics": {
        "r2": metrics['r2'],
        "mse": metrics['mse'],
        "pcc": metrics['pcc'],
        "r2_de": metrics['r2_de'],
        "mse_de": metrics['mse_de'],
        "pcc_de": metrics['pcc_de']
    },
    "nmi": None,
    "ari": None,
    "asw_celltype": None,
    "asw_batch": None,
    "knn_cross": None
}
