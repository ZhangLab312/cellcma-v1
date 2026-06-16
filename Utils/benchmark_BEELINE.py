from pathlib import Path
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")  # Globally ignore all warnings
# Default path, can be overridden by function parameter or CHIPSEQ_GT_PATH env var
ground_truth_path = os.environ.get(
    "CHIPSEQ_GT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset", "chip_seq_TF"),
)


def calculate_overall_metrics(df):
    """Calculate overall evaluation metrics: mean of each metric across all valid TFs."""
    metric_cols = ['AUC', 'AUPR', 'AUPRC Ratio', 'F1', 'G-Mean', 'Accuracy', 'BACC']
    overall_metrics = []
    for col in metric_cols:
        valid = df[col].replace(0, np.nan).dropna()
        overall_metrics.append({
            'Metric': col,
            'Mean': round(valid.mean(), 4),
            'Valid_TF_Count': len(valid),
        })
    return pd.DataFrame(overall_metrics)


def benchmark_expand(adj_file, isFromFile, top_k=1, print_each_TF=False, save_true_edges=False,
                     method="GRMI-Net", gt_path=None):
    """Evaluate GRN predictions against ChIP-seq ground truth.

    Args:
        adj_file: DataFrame or path to CSV with columns [TF, Target, EdgeWeight]
        isFromFile: if True, adj_file is a file path; if False, it's a DataFrame
        top_k: fraction of top edges to keep (0~1)
        print_each_TF: print per-TF metrics
        save_true_edges: save true positive edges to file
        method: method name for output file naming
        gt_path: override ground truth directory (default: global ground_truth_path)
    """
    if gt_path is not None:
        _gt_path = gt_path
    else:
        _gt_path = ground_truth_path
    if isFromFile:
        adj_final = pd.read_csv(adj_file, index_col=0)
    else:
        adj_final = adj_file
    adj_final = adj_final.head(int(len(adj_final) * top_k))
    adj_final = adj_final.query("TF!=Target")
    # adj_final["EdgeWeight"] = adj_final["EdgeWeight"]*10000
    # adj_final = adj_final[adj_final["EdgeWeight"] >= 8e-4]
    TF_list = adj_final["TF"].unique().tolist()
    results = []

    for TF_name in TF_list:
        print(TF_name)
        ground_truth_file_path = os.path.join(_gt_path, f"{TF_name}.CSV")
        file_path = Path(ground_truth_file_path)
        if not file_path.is_file():
            continue
        try:
            ground_truth_file = pd.read_csv(ground_truth_file_path, index_col=0)
        except Exception as e:
            print(e)
            print(TF_name)
            continue

        # ground_truth_file = ground_truth_file.query(f"TF=='{TF_name}'")
        label = ground_truth_file["TG"].values.tolist()
        TF_adj = adj_final.query(f"TF=='{TF_name}'")
        if len(TF_adj) < 1:
            # print(f"{TF_name}:data shorten")
            continue

        pos_ratio = len(label) / len(ground_truth_file) if len(ground_truth_file) > 0 else 0
        target_gene = TF_adj["Target"].values.tolist()
        Score = TF_adj["EdgeWeight"].values.tolist()

        TGset = target_gene.copy()
        d1 = np.zeros(len(TGset))
        loc = np.where(np.isin(TGset, label))[0]
        d1[loc] = 1

        try:
            # ROC-related metrics
            fpr, tpr, thresholds = roc_curve(d1, Score)
            auc_score = roc_auc_score(d1, Score)
            # AUROC = auc(fpr,tpr)
            aupr = average_precision_score(d1, Score)
            aupr_ratio = aupr / np.mean(d1) if pos_ratio > 0 else np.nan

            # Calculate optimal threshold
            precision, recall, pr_thresholds = precision_recall_curve(d1, Score)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            max_f1_idx = np.argmax(f1_scores)
            max_f1 = f1_scores[max_f1_idx]

            # Calculate accuracy using the optimal threshold
            best_threshold = pr_thresholds[max_f1_idx]
            y_pred = (np.array(Score) > best_threshold).astype(int)
            true_positives_mask = (y_pred == 1) & (d1 == 1)
            true_edges = TF_adj[true_positives_mask].copy()
            true_edges['Actual_Label'] = d1[true_positives_mask]  # Add actual label column
            if save_true_edges:
                # Save true edges for each TF
                true_edges_path = f"../output/true_edges/{method}/{TF_name}_true_edges.csv"
                os.makedirs(os.path.dirname(true_edges_path), exist_ok=True)
                true_edges.to_csv(true_edges_path, index=False)
            accuracy = np.mean(y_pred == d1)  # Accuracy formula
            gmeans = np.sqrt(tpr * (1 - fpr))  # G-Mean = sqrt(TPR * TNR)
            best_gmean_idx = np.argmax(gmeans)
            best_gmean_threshold = thresholds[best_gmean_idx]
            y_pred_gmean = (Score >= best_gmean_threshold).astype(int)
            # Calculate G-Mean
            gmean = geometric_mean_score(d1, y_pred_gmean)
            balanced_acc = balanced_accuracy_score(d1, y_pred_gmean)

        except Exception as e:
            # print(f"Error calculating metrics for {TF_name}: {str(e)}")
            continue

        # Store results (including Accuracy)
        results.append({
            "TF": TF_name,
            "AUC": round(auc_score, 4),
            # "AUROC": round(AUROC, 4),
            "AUPR": round(aupr, 4),
            "AUPRC Ratio": round(aupr_ratio, 4) if not np.isnan(aupr_ratio) else np.nan,
            "F1": round(max_f1, 4),
            "G-Mean": round(gmean, 4),
            "Accuracy": round(accuracy, 4),  # Added accuracy
            "BACC": round(balanced_acc, 4),
        })

        if print_each_TF:
            print(
                f"{TF_name}: AUC:{auc_score:.4f}, AUPR:{aupr:.4f}, Ratio:{aupr_ratio:.4f}, F1:{max_f1:.4f}, G-Mean:{gmean:.4f}, Accuracy:{accuracy:.4f},num:{len(TF_adj)}")

    # Save results (including Accuracy)
    output_path = "../output/evaluation_metrics_rna_piror.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # Calculate and save overall metrics (including Accuracy)
    overall_df = calculate_overall_metrics(results_df)
    print("\nOverall evaluation metrics:")
    print(overall_df.to_string(index=False))

    overall_output_path = "../output/overall_metrics_rna_piror.csv"
    os.makedirs(os.path.dirname(overall_output_path), exist_ok=True)
    overall_df.to_csv(overall_output_path, index=False)

    return overall_df

def benchmark(adj_file,save_true_edges, top_k=1):
    return benchmark_expand(adj_file, isFromFile=True, top_k=top_k,print_each_TF=True,save_true_edges=save_true_edges)


if __name__ == "__main__":
    # aupr =0.58(2e-3) 0.6086(8e-4) 0.2501(3e-4)
    # benchmark("../output/test_rna_atac_best_2025-03-20_10-58-49.csv",top_k=1)
    # aupr = 0.6033(2e-3) 0.7542(8e-4) 0.2433(3e-4)
    # benchmark("../output/test_rna_atac_best_2025-03-21_07-12-35.csv",top_k=1,save_true_edges=False)
    # aupr = 0.5042(2e-3) 0.7730(8e-4) 0.2443(3e-4)
    # benchmark("../output/test_rna_atac_best_2025-03-23_21-09-23.csv",top_k=1,save_true_edges=False)
    # aupr = 0.5659(2e-3) 0.5665(8e-4) 0.5221(3e-4)
    # benchmark("../output/test_rna_atac_best_2025-03-29_10-20-44.csv",top_k=1,save_true_edges=False)
    # aupr = 0.6264(2e-3) 0.6264(8e-4) 0.68(3e-4)
    # benchmark("../output/test_rna_atac_best_2025-04-19_16-22-23.csv",top_k=1,save_true_edges=False)
    # benchmark("../output/test_BEELINE_best_2025-05-06_11-27-38.csv",top_k=1)
    # 0.5158
    # benchmark("../output/test_rna_atac_best_2025-05-11_21-29-19.csv",top_k=1,save_true_edges=False)
    # PBMC GRMI-net
    # benchmark("../output/test_rna_atac_best_2025-05-10_23-08-07.csv",top_k=1,save_true_edges=False)
    # adj_final = pd.read_csv("../data/GRN_inference/output/500_ChIP-seq_hESC_demo_output.tsv",sep="\t")
    # adj_final.columns = ["TF","Target","EdgeWeight"]
    #
    # benchmark_expand(adj_final, isFromFile=False, top_k=1, print_each_TF=True)
    # adj_final = pd.read_csv("../compare/draft_grn.csv")
    # adj_final.columns = ["TF","Target","EdgeWeight"]
    #
    # benchmark_expand(adj_final, isFromFile=False, top_k=0.50, print_each_TF=True)
    # glue
    # adj_final = pd.read_csv("../compare/glue_grn.csv",index_col=0,header=0)
    # adj_final.columns = ["TF","Target","EdgeWeight"]
    # # linger
    # adj_final = pd.read_csv("../compare/linger_grn.csv",header=0)
    # adj_final.columns = ["TF","Target","EdgeWeight"]
    #
    # adj_final = pd.read_csv("../output/GRN_inference_result_Buen.tsv",header=0,sep="\t")
    # adj_final.columns = ["TF","Target","EdgeWeight"]
    # benchmark_expand(adj_final, isFromFile=False, top_k=1, print_each_TF=True,save_true_edges=False,method="glue")
    # scGPT Buen
    # adj_final = pd.read_csv("../compare/scGPT/PBMC/pbmc_gene_pairs_weighted_bidirectional.csv", header=0)
    # # adj_final["EdgeWeight"]=0.6
    # adj_final.columns = ["TF","Target","EdgeWeight"]

    # adj_final = pd.read_csv("../output/TF_target_edge_list_unpairreg.csv", header=0)
    # adj_final.columns = ["TF","Target","EdgeWeight"]
    # # adj_final["EdgeWeight"]=0.6
    # benchmark_expand(adj_final, isFromFile=False, top_k=1, print_each_TF=True,save_true_edges=False,method="glue")

    # benchmark("../output/TF_target_edge_list_unpairreg.csv",top_k=1,save_true_edges=False)
    adj_final = pd.read_csv("../output/unpairReg_PBMC_tf.csv", header=0,index_col=0)
    adj_final.columns = ["TF","Target","EdgeWeight"]
    adj_final1 = adj_final.copy()
    adj_final1=adj_final1[["Target","TF","EdgeWeight"]]
    adj_final1.columns = ["TF","Target","EdgeWeight"]
    adj_final = df_concat = pd.concat([adj_final, adj_final1], ignore_index=True)
    # benchmark("../output/TF_target_edge_list_unpairreg.csv",top_k=1,save_true_edges=False)
    benchmark_expand(adj_final, isFromFile=False, top_k=1, print_each_TF=True,save_true_edges=False,method="glue")

