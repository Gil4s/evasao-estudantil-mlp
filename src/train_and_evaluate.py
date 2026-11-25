"""Complete neural network training, validation, ablation, interpretability, and figure generation pipeline.

Follows strict ML practices:
- Stratified 70% Train / 15% Validation / 15% Test split BEFORE any preprocessing.
- ColumnTransformer (OneHotEncoder for nominal codes + StandardScaler for continuous features) fit ONLY on Train.
- Unbiased model selection on 3-seed mean Validation ROC-AUC (zero test snooping).
- Compares Dummy, Perceptron (0 hidden layers), LogisticRegression, and PyTorch MLP (1, 2, 3 hidden layers).
- Includes Early-Detection analysis (Enrollment -> 1st Sem -> 2nd Sem), all-student binary & 3-class robustness,
  grouped permutation importance, subgroup slice analysis, and 5 publication-grade figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BINARY_COLS,
    FIGURES_DIR,
    NOMINAL_CATEGORICAL_COLS,
    PROCESSED_DIR,
    REPORTS_DIR,
    SEED,
    STAGE_ENROLLMENT_COLS,
    STAGE_FULL_COLS,
    STAGE_SEM1_COLS,
    TEST_SIZE,
    VAL_SIZE,
)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "font.sans-serif": ["DejaVu Sans", "Arial"],
    }
)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """One-hot encode nominal categorical codes (restricting rare tails) and standardize numerical features."""
    cat_cols = [c for c in feature_cols if c in NOMINAL_CATEGORICAL_COLS]
    bin_cols = [c for c in feature_cols if c in BINARY_COLS]
    num_cols = [c for c in feature_cols if c not in cat_cols and c not in bin_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


class StudentDropoutMLP(nn.Module):
    """Configurable feedforward MLP with ReLU and Dropout for student dropout prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.30,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            drop_p = dropout if i == 0 else max(0.15, dropout - 0.10)
            layers.append(nn.Dropout(drop_p))
            in_d = h
        layers.append(nn.Linear(in_d, num_outputs))
        self.net = nn.Sequential(*layers)
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.num_outputs == 1 else out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_binary_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: tuple[int, ...] = (64, 32),
    dropout: float = 0.30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 64,
    max_epochs: int = 120,
    patience: int = 18,
    seed: int = SEED,
) -> tuple[StudentDropoutMLP, dict]:
    """Train binary PyTorch MLP with AdamW and Early Stopping on Validation ROC-AUC."""
    set_seed(seed)
    model = StudentDropoutMLP(X_tr.shape[1], hidden_layers=hidden_layers, dropout=dropout, num_outputs=1)

    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)

        train_loss = total_loss / len(X_tr)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_probs = torch.sigmoid(val_logits).numpy()
        val_auc = float(roc_auc_score(y_val, val_probs))

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_auc"].append(round(val_auc, 4))

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "hidden_layers": list(hidden_layers),
        "num_trainable_layers": len(hidden_layers) + 1,
        "input_dim_after_encoding": int(X_tr.shape[1]),
        "num_parameters": model.count_parameters(),
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
        "best_val_auc": round(best_auc, 4),
        "train_time_sec": round(time.perf_counter() - t0, 2),
        "history": history,
    }
    return model, info


def predict_proba_binary(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def tune_threshold_f1(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.30, 0.70, 41):
        s = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        if s > best_f1:
            best_f1 = s
            best_t = float(t)
    return best_t


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 800) -> list[float]:
    rng = np.random.default_rng(SEED)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "auc_ci95": bootstrap_auc_ci(y_true, scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_multiclass_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: tuple[int, ...] = (64, 32),
) -> tuple[StudentDropoutMLP, float]:
    """Train 3-class PyTorch MLP (Dropout, Enrolled, Graduate) as complementary analysis."""
    set_seed(SEED)
    model = StudentDropoutMLP(X_tr.shape[1], hidden_layers=hidden_layers, dropout=0.30, num_outputs=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=64,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1).numpy()
        val_f1 = float(f1_score(y_val, preds, average="macro"))
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 15:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, round(best_f1, 4)


def grouped_permutation_importance(
    model: nn.Module,
    prep: ColumnTransformer,
    df_te: pd.DataFrame,
    y_te: np.ndarray,
    raw_cols: list[str],
    n_repeats: int = 12,
) -> pd.DataFrame:
    """Shuffle each ORIGINAL raw column before preprocessing so categorical dummies stay grouped."""
    rng = np.random.default_rng(SEED)
    base_probs = predict_proba_binary(model, prep.transform(df_te[raw_cols]))
    base_auc = roc_auc_score(y_te, base_probs)

    pretty_labels = {
        "cu_2nd_approved": "Disciplinas aprovadas (2º sem.)",
        "cu_1st_approved": "Disciplinas aprovadas (1º sem.)",
        "approval_rate_2nd_sem": "Taxa de aprovação (2º sem.)",
        "approval_rate_1st_sem": "Taxa de aprovação (1º sem.)",
        "tuition_fees_up_to_date": "Mensalidades em dia",
        "cu_2nd_grade": "Nota média (2º sem.)",
        "cu_1st_grade": "Nota média (1º sem.)",
        "cu_2nd_enrolled": "Disciplinas matriculadas (2º sem.)",
        "cu_1st_enrolled": "Disciplinas matriculadas (1º sem.)",
        "cu_2nd_credited": "Disciplinas aproveitadas (2º sem.)",
        "cu_1st_credited": "Disciplinas aproveitadas (1º sem.)",
        "cu_2nd_evaluations": "Avaliações realizadas (2º sem.)",
        "cu_1st_evaluations": "Avaliações realizadas (1º sem.)",
        "cu_2nd_without_eval": "Disciplinas sem avaliação (2º sem.)",
        "cu_1st_without_eval": "Disciplinas sem avaliação (1º sem.)",
        "scholarship_holder": "Bolsista",
        "age_at_enrollment": "Idade na matrícula",
        "course": "Curso de graduação",
        "debtor": "Inadimplente (devedor)",
        "admission_grade": "Nota de ingresso",
        "application_mode": "Modalidade de ingresso",
        "application_order": "Ordem de opção do curso",
        "previous_qualification": "Formação anterior",
        "previous_qualification_grade": "Nota da formação anterior",
        "mothers_occupation": "Ocupação da mãe",
        "mothers_qualification": "Escolaridade da mãe",
        "fathers_occupation": "Ocupação do pai",
        "fathers_qualification": "Escolaridade do pai",
        "marital_status": "Estado civil",
        "daytime_attendance": "Turno diurno",
        "displaced": "Estudante deslocado",
        "gender": "Gênero",
        "international": "Estudante internacional",
        "educational_special_needs": "Necessidades educacionais especiais",
        "unemployment_rate": "Taxa de desemprego",
        "inflation_rate": "Taxa de inflação",
        "gdp": "Variação do PIB",
        "grade_delta_2nd_vs_1st": "Variação de nota (2º vs 1º sem.)",
        "approval_delta_2nd_vs_1st": "Variação de aprovação (2º vs 1º sem.)",
    }

    rows = []
    for col in raw_cols:
        drops = []
        df_perm = df_te[raw_cols].copy()
        orig_vals = df_perm[col].values.copy()
        for _ in range(n_repeats):
            df_perm[col] = rng.permutation(orig_vals)
            perm_auc = roc_auc_score(y_te, predict_proba_binary(model, prep.transform(df_perm)))
            drops.append(base_auc - perm_auc)
        df_perm[col] = orig_vals

        if col in ("cu_1st_credited", "cu_1st_enrolled", "cu_1st_evaluations", "cu_1st_approved", "cu_1st_grade", "cu_1st_without_eval", "approval_rate_1st_sem"):
            grp = "Desempenho 1º Semestre"
        elif col in ("cu_2nd_credited", "cu_2nd_enrolled", "cu_2nd_evaluations", "cu_2nd_approved", "cu_2nd_grade", "cu_2nd_without_eval", "approval_rate_2nd_sem", "grade_delta_2nd_vs_1st", "approval_delta_2nd_vs_1st"):
            grp = "Desempenho 2º Semestre"
        elif col in ("tuition_fees_up_to_date", "debtor", "scholarship_holder"):
            grp = "Financeiro / Bolsa"
        else:
            grp = "Perfil / Ingresso / Macro"

        rows.append(
            {
                "feature": col,
                "label_pt": pretty_labels.get(col, col),
                "group": grp,
                "delta_auc_mean": float(np.mean(drops)),
                "delta_auc_std": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_auc_mean", ascending=False).reset_index(drop=True)


# --- Plotting helpers (publication-grade PT-BR charts) ---

def plot_eda_and_class_balance(df_all: pd.DataFrame, df_bin: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # 1. Class distribution across all 4,424 students
    counts = df_all["target"].value_counts().reindex(["Graduate", "Dropout", "Enrolled"])
    labels_pt = ["Formado\n(Graduate)", "Evadido\n(Dropout)", "Matriculado\n(Enrolled)"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    bars = axes[0].bar(labels_pt, counts.values, color=colors, alpha=0.9)
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 45, f"{v:,}\n({v/len(df_all)*100:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[0].set_ylim(0, max(counts.values) * 1.18)
    axes[0].set_ylabel("Número de Estudantes")
    axes[0].set_title("Distribuição das Classes (n = 4.424)")

    # 2. Dropout rate by key financial/academic factors
    factors = [
        ("Mensalidade\nAtrasada", df_bin.loc[df_bin["tuition_fees_up_to_date"] == 0, "target_dropout"].mean() * 100),
        ("Devedor\n(Debtor)", df_bin.loc[df_bin["debtor"] == 1, "target_dropout"].mean() * 100),
        ("Turno\nNoturno", df_bin.loc[df_bin["daytime_attendance"] == 0, "target_dropout"].mean() * 100),
        ("Média Geral\n(3.630)", df_bin["target_dropout"].mean() * 100),
        ("Mensalidade\nem Dia", df_bin.loc[df_bin["tuition_fees_up_to_date"] == 1, "target_dropout"].mean() * 100),
        ("Bolsista\n(Scholarship)", df_bin.loc[df_bin["scholarship_holder"] == 1, "target_dropout"].mean() * 100),
    ]
    f_names = [f[0] for f in factors]
    f_vals = [f[1] for f in factors]
    bar_cols = ["#d62728", "#e6550d", "#ff7f0e", "#4c72b0", "#55a868", "#2ca02c"]
    b2 = axes[1].barh(f_names[::-1], f_vals[::-1], color=bar_cols[::-1], alpha=0.9)
    for b, v in zip(b2, f_vals[::-1]):
        axes[1].text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Taxa de Evasão (%)")
    axes[1].set_title("Taxa de Evasão por Perfil (Desfecho Final)")

    # 3. 1st & 2nd semester approval rate distribution: Dropout vs Graduate
    plot_df = pd.DataFrame(
        {
            "Semestre": ["1º Semestre"] * len(df_bin) + ["2º Semestre"] * len(df_bin),
            "Taxa de Aprovação (%)": np.concatenate(
                [df_bin["approval_rate_1st_sem"].values * 100, df_bin["approval_rate_2nd_sem"].values * 100]
            ),
            "Desfecho": np.concatenate(
                [
                    np.where(df_bin["target_dropout"] == 1, "Evadido", "Formado"),
                    np.where(df_bin["target_dropout"] == 1, "Evadido", "Formado"),
                ]
            ),
        }
    )
    sns.boxplot(
        data=plot_df,
        x="Semestre",
        y="Taxa de Aprovação (%)",
        hue="Desfecho",
        palette={"Formado": "#2ca02c", "Evadido": "#d62728"},
        ax=axes[2],
        width=0.55,
    )
    axes[2].set_title("Aprovação Acadêmica: Evadido vs. Formado")
    axes[2].legend(title="", loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, best_epoch: int, arch_label: str, out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    axes[0].plot(epochs, history["train_loss"], label="Perda Treino (BCE ponderada)", linewidth=2.1, color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], label="Perda Validação", linewidth=2.1, color="#ff7f0e")
    axes[0].axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.4, label=f"Melhor época ({best_epoch})")
    axes[0].set_title(f"Curva de Perda — PyTorch {arch_label}")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(frameon=True)

    axes[1].plot(epochs, history["val_auc"], label="ROC-AUC Validação", linewidth=2.2, color="#2ca02c")
    axes[1].axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.4, label=f"Early Stopping (época {best_epoch})")
    axes[1].scatter([best_epoch], [history["val_auc"][best_epoch - 1]], color="#d62728", zorder=5, s=55)
    axes[1].set_title("Evolução do ROC-AUC na Validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_and_confusion(
    y_te: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    perc_cm: np.ndarray,
    mlp_cm: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    colors = {
        "Perceptron (0 ocultas)": "#d62728",
        "Regressão Logística": "#ff7f0e",
        "PyTorch MLP (selecionada)": "#1f77b4",
    }
    for label, sc in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_te, sc)
        auc = roc_auc_score(y_te, sc)
        ax.plot(fpr, tpr, linewidth=2.3, label=f"{label} (AUC={auc:.3f})", color=colors.get(label))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (AUC=0.500)")
    ax.set_title("Curva ROC — Teste Hold-Out (n = 545)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.legend(loc="lower right", fontsize=8.3, frameon=True)

    for ax_cm, cm, title in [
        (axes[1], perc_cm, "Matriz de Confusão — Perceptron"),
        (axes[2], mlp_cm, "Matriz de Confusão — PyTorch MLP"),
    ]:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Formado", "Evadido"],
            yticklabels=["Formado", "Evadido"],
            ax=ax_cm,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax_cm.set_title(title)
        ax_cm.set_xlabel("Previsto")
        ax_cm.set_ylabel("Real")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_and_early_detection(
    arch_rows: list[dict],
    early_rows: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    # Left: Architecture ablation (Val vs Test ROC-AUC)
    df_a = pd.DataFrame(arch_rows)
    x = np.arange(len(df_a))
    w = 0.36
    axes[0].bar(x - w / 2, df_a["val_auc"], width=w, label="Validação (3 seeds)", color="#4c72b0")
    axes[0].bar(x + w / 2, df_a["test_auc"], width=w, label="Teste final (hold-out)", color="#55a868")
    for i, r in df_a.iterrows():
        axes[0].text(i - w / 2, r["val_auc"] + 0.005, f"{r['val_auc']:.3f}", ha="center", va="bottom", fontsize=8)
        axes[0].text(i + w / 2, r["test_auc"] + 0.005, f"{r['test_auc']:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_a["label"], rotation=18, ha="right", fontsize=8.5)
    axes[0].set_ylim(0.84, 0.995)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Ablação de Arquitetura: 0, 1, 2 e 3 Camadas Ocultas")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=True)

    # Right: Early detection stages (Enrollment -> 1st Sem -> 2nd Sem)
    df_e = pd.DataFrame(early_rows)
    x2 = np.arange(len(df_e))
    w2 = 0.26
    axes[1].bar(x2 - w2, df_e["accuracy"], width=w2, label="Acurácia", color="#4c72b0")
    axes[1].bar(x2, df_e["f1"], width=w2, label="F1-Score", color="#dd8452")
    axes[1].bar(x2 + w2, df_e["roc_auc"], width=w2, label="ROC-AUC", color="#2ca02c")
    for i, r in df_e.iterrows():
        axes[1].text(i - w2, r["accuracy"] + 0.012, f"{r['accuracy']:.2f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(i, r["f1"] + 0.012, f"{r['f1']:.2f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(i + w2, r["roc_auc"] + 0.012, f"{r['roc_auc']:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(df_e["stage_label"], fontsize=9)
    axes[1].set_ylim(0.50, 1.04)
    axes[1].set_ylabel("Desempenho no Teste (PyTorch MLP)")
    axes[1].set_title("Detecção Precoce: Matrícula → 1º Semestre → 2º Semestre")
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_importance_and_slices(imp_df: pd.DataFrame, slice_dict: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # Left: Top 12 permutation feature importance
    top12 = imp_df.head(12).iloc[::-1]
    palette = {
        "Desempenho 2º Semestre": "#1f77b4",
        "Desempenho 1º Semestre": "#2ca02c",
        "Financeiro / Bolsa": "#d62728",
        "Perfil / Ingresso / Macro": "#9467bd",
    }
    colors = [palette[g] for g in top12["group"]]
    axes[0].barh(top12["label_pt"], top12["delta_auc_mean"], xerr=top12["delta_auc_std"], color=colors, alpha=0.9, capsize=3)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlim(0, max(top12["delta_auc_mean"] + top12["delta_auc_std"]) * 1.12)
    axes[0].set_xlabel("Queda no ROC-AUC ao embaralhar a variável (Δ AUC)")
    axes[0].set_title("Top 12 Variáveis Mais Importantes na PyTorch MLP")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=lbl) for lbl, c in palette.items()]
    axes[0].legend(handles=handles, loc="lower right", frameon=True, fontsize=8)

    # Right: Subgroup slice performance (Accuracy & ROC-AUC)
    s_labels = list(slice_dict.keys())
    s_acc = [slice_dict[k]["accuracy"] for k in s_labels]
    s_auc = [slice_dict[k]["roc_auc"] for k in s_labels]
    y_pos = np.arange(len(s_labels))
    h = 0.36
    axes[1].barh(y_pos + h / 2, s_auc, height=h, label="ROC-AUC", color="#2ca02c", alpha=0.9)
    axes[1].barh(y_pos - h / 2, s_acc, height=h, label="Acurácia", color="#4c72b0", alpha=0.9)
    for i in range(len(s_labels)):
        axes[1].text(s_auc[i] + 0.006, y_pos[i] + h / 2, f"{s_auc[i]:.3f}", va="center", fontsize=8.3, fontweight="bold")
        axes[1].text(s_acc[i] + 0.006, y_pos[i] - h / 2, f"{s_acc[i]:.3f}", va="center", fontsize=8.3)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{k}\n(n={slice_dict[k]['rows']})" for k in s_labels], fontsize=8.5)
    axes[1].set_xlim(0.65, 1.07)
    axes[1].set_xlabel("Métrica no Teste")
    axes[1].set_title("Consistência da MLP por Subgrupo de Estudantes")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_pipeline() -> dict:
    """Execute full experimental pipeline and export JSON summary + 5 charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(PROCESSED_DIR / "student_dropout_all.csv")
    df_bin = pd.read_csv(PROCESSED_DIR / "student_dropout_binary.csv")

    # --- Stratified 70% Train / 15% Val / 15% Test split on primary task (Dropout vs Graduate, n=3,630) ---
    idx_all = np.arange(len(df_bin))
    y_bin = df_bin["target_dropout"].values
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=VAL_SIZE + TEST_SIZE, stratify=y_bin, random_state=SEED)
    idx_va, idx_te = train_test_split(
        idx_tmp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=y_bin[idx_tmp],
        random_state=SEED,
    )

    df_tr, df_va, df_te = df_bin.iloc[idx_tr], df_bin.iloc[idx_va], df_bin.iloc[idx_te]
    y_tr, y_va, y_te = y_bin[idx_tr], y_bin[idx_va], y_bin[idx_te]

    # Fit preprocessing strictly on Train
    prep_full = build_preprocessor(STAGE_FULL_COLS)
    X_tr = prep_full.fit_transform(df_tr[STAGE_FULL_COLS])
    X_va = prep_full.transform(df_va[STAGE_FULL_COLS])
    X_te = prep_full.transform(df_te[STAGE_FULL_COLS])

    # --- 1. Baselines ---
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    dummy_m = {"accuracy": round(float(accuracy_score(y_te, dummy.predict(X_te))), 4), "f1": 0.0, "roc_auc": 0.5}

    perc = Perceptron(max_iter=1000, class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
    perc_va_auc = round(float(roc_auc_score(y_va, perc.decision_function(X_va))), 4)
    perc_te_scores = perc.decision_function(X_te)
    perc_te_pred = perc.predict(X_te)
    perc_m = eval_binary(y_te, perc_te_pred, perc_te_scores)
    perc_m["val_auc"] = perc_va_auc
    perc_m["num_parameters"] = int(X_tr.shape[1] + 1)

    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
    lr_va_probs = lr.predict_proba(X_va)[:, 1]
    lr_thr = tune_threshold_f1(y_va, lr_va_probs)
    lr_te_probs = lr.predict_proba(X_te)[:, 1]
    lr_m = eval_binary(y_te, (lr_te_probs >= lr_thr).astype(int), lr_te_probs)
    lr_m["val_auc"] = round(float(roc_auc_score(y_va, lr_va_probs)), 4)
    lr_m["val_threshold"] = round(lr_thr, 2)
    lr_m["num_parameters"] = int(X_tr.shape[1] + 1)

    # --- 2. Architecture Ablation (strictly chosen by 3-seed mean Validation ROC-AUC) ---
    arch_candidates = [
        ("MLP 1 oculta (32)", (32,)),
        ("MLP 1 oculta (64)", (64,)),
        ("MLP 2 ocultas (64→32)", (64, 32)),
        ("MLP 3 ocultas (128→64→32)", (128, 64, 32)),
    ]
    arch_rows = [
        {
            "label": "Perceptron (0 ocultas)",
            "hidden_layers": [],
            "num_trainable_layers": 1,
            "num_parameters": perc_m["num_parameters"],
            "val_auc": perc_va_auc,
            "test_auc": perc_m["roc_auc"],
            "test_f1": perc_m["f1"],
            "test_acc": perc_m["accuracy"],
        }
    ]
    trained_models = {}
    val_seeds = (42, 102, 202)

    for label, hidden in arch_candidates:
        seed_aucs = []
        primary = None
        for idx_s, s in enumerate(val_seeds):
            m_s, info_s = train_binary_mlp(X_tr, y_tr, X_va, y_va, hidden_layers=hidden, seed=s)
            seed_aucs.append(info_s["best_val_auc"])
            if idx_s == 0:
                primary = (m_s, info_s)
        model, info = primary
        mean_val_auc = round(float(np.mean(seed_aucs)), 4)
        std_val_auc = round(float(np.std(seed_aucs)), 4)
        va_probs = predict_proba_binary(model, X_va)
        te_probs = predict_proba_binary(model, X_te)
        thr = tune_threshold_f1(y_va, va_probs)
        te_m = eval_binary(y_te, (te_probs >= thr).astype(int), te_probs)

        arch_rows.append(
            {
                "label": label,
                "hidden_layers": list(hidden),
                "num_trainable_layers": info["num_trainable_layers"],
                "num_parameters": info["num_parameters"],
                "best_epoch": info["best_epoch"],
                "train_time_sec": info["train_time_sec"],
                "val_auc": mean_val_auc,
                "val_auc_std": std_val_auc,
                "val_threshold": round(thr, 2),
                "test_auc": te_m["roc_auc"],
                "test_f1": te_m["f1"],
                "test_acc": te_m["accuracy"],
            }
        )
        trained_models[label] = (model, info, thr, te_probs, te_m)

    mlp_only = [r for r in arch_rows if r["num_trainable_layers"] > 1]
    best_arch_row = max(mlp_only, key=lambda r: r["val_auc"])
    best_label = best_arch_row["label"]
    best_hidden = tuple(best_arch_row["hidden_layers"])
    best_model, best_info, best_thr, best_te_probs, best_te_m = trained_models[best_label]
    best_te_pred = (best_te_probs >= best_thr).astype(int)

    # --- 3. Early Detection Analysis (Enrollment -> 1st Sem -> 2nd Sem) using selected architecture ---
    early_rows = []
    for stage_label, cols in [
        ("Na Matrícula\n(Dia 1)", STAGE_ENROLLMENT_COLS),
        ("Após 1º Semestre\n(Intervenção Precoce)", STAGE_SEM1_COLS),
        ("Após 2º Semestre\n(Modelo Completo)", STAGE_FULL_COLS),
    ]:
        p_st = build_preprocessor(cols)
        Xs_tr = p_st.fit_transform(df_tr[cols])
        Xs_va = p_st.transform(df_va[cols])
        Xs_te = p_st.transform(df_te[cols])
        m_st, info_st = train_binary_mlp(Xs_tr, y_tr, Xs_va, y_va, hidden_layers=best_hidden)
        t_st = tune_threshold_f1(y_va, predict_proba_binary(m_st, Xs_va))
        pr_st = predict_proba_binary(m_st, Xs_te)
        em = eval_binary(y_te, (pr_st >= t_st).astype(int), pr_st)
        early_rows.append(
            {
                "stage_label": stage_label,
                "stage_clean": stage_label.replace("\n", " "),
                "raw_cols_count": len(cols),
                "encoded_dim": int(Xs_tr.shape[1]),
                "val_auc": info_st["best_val_auc"],
                **em,
            }
        )

    # --- 4. Complementary Formulations on ALL 4,424 Students ---
    # 4A) Binary: Dropout vs Non-Dropout (4,424 students)
    y_all_bin = df_all["target_dropout"].values
    ia_tr, ia_tmp = train_test_split(np.arange(len(df_all)), test_size=0.30, stratify=y_all_bin, random_state=SEED)
    ia_va, ia_te = train_test_split(ia_tmp, test_size=0.50, stratify=y_all_bin[ia_tmp], random_state=SEED)
    p_all = build_preprocessor(STAGE_FULL_COLS)
    Xa_tr = p_all.fit_transform(df_all.iloc[ia_tr][STAGE_FULL_COLS])
    Xa_va = p_all.transform(df_all.iloc[ia_va][STAGE_FULL_COLS])
    Xa_te = p_all.transform(df_all.iloc[ia_te][STAGE_FULL_COLS])

    perc_all = Perceptron(max_iter=1000, class_weight="balanced", random_state=SEED).fit(Xa_tr, y_all_bin[ia_tr])
    m_all, _ = train_binary_mlp(Xa_tr, y_all_bin[ia_tr], Xa_va, y_all_bin[ia_va], hidden_layers=best_hidden)
    t_all = tune_threshold_f1(y_all_bin[ia_va], predict_proba_binary(m_all, Xa_va))
    p_all_te = predict_proba_binary(m_all, Xa_te)
    all_students_binary = {
        "n_total": int(len(df_all)),
        "n_test": int(len(ia_te)),
        "perceptron": eval_binary(y_all_bin[ia_te], perc_all.predict(Xa_te), perc_all.decision_function(Xa_te)),
        "pytorch_mlp": eval_binary(y_all_bin[ia_te], (p_all_te >= t_all).astype(int), p_all_te),
    }

    # 4B) Multiclass (3 classes: Dropout=0, Enrolled=1, Graduate=2)
    cls_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    y_mc = df_all["target"].map(cls_map).values
    im_tr, im_tmp = train_test_split(np.arange(len(df_all)), test_size=0.30, stratify=y_mc, random_state=SEED)
    im_va, im_te = train_test_split(im_tmp, test_size=0.50, stratify=y_mc[im_tmp], random_state=SEED)
    p_mc = build_preprocessor(STAGE_FULL_COLS)
    Xm_tr = p_mc.fit_transform(df_all.iloc[im_tr][STAGE_FULL_COLS])
    Xm_va = p_mc.transform(df_all.iloc[im_va][STAGE_FULL_COLS])
    Xm_te = p_mc.transform(df_all.iloc[im_te][STAGE_FULL_COLS])
    m_mc, mc_val_f1 = train_multiclass_mlp(Xm_tr, y_mc[im_tr], Xm_va, y_mc[im_va], hidden_layers=best_hidden)
    m_mc.eval()
    with torch.no_grad():
        mc_preds = m_mc(torch.tensor(Xm_te, dtype=torch.float32)).argmax(dim=1).numpy()
    per_class_f1 = f1_score(y_mc[im_te], mc_preds, average=None)
    multiclass_3 = {
        "n_test": int(len(im_te)),
        "val_macro_f1": mc_val_f1,
        "test_accuracy": round(float(accuracy_score(y_mc[im_te], mc_preds)), 4),
        "test_macro_f1": round(float(f1_score(y_mc[im_te], mc_preds, average="macro")), 4),
        "f1_per_class": {
            "Dropout": round(float(per_class_f1[0]), 4),
            "Enrolled": round(float(per_class_f1[1]), 4),
            "Graduate": round(float(per_class_f1[2]), 4),
        },
        "confusion_matrix_dropout_enrolled_graduate": confusion_matrix(y_mc[im_te], mc_preds).tolist(),
    }

    # --- 5. Grouped Permutation Importance & Subgroup Slice Analysis ---
    imp_df = grouped_permutation_importance(best_model, prep_full, df_te, y_te, STAGE_FULL_COLS, n_repeats=12)

    slice_defs = {
        "Bolsista": df_te["scholarship_holder"].values == 1,
        "Não-Bolsista": df_te["scholarship_holder"].values == 0,
        "Turno Diurno": df_te["daytime_attendance"].values == 1,
        "Turno Noturno": df_te["daytime_attendance"].values == 0,
        "Idade ≤ 20 anos": df_te["age_at_enrollment"].values <= 20,
        "Idade > 20 anos": df_te["age_at_enrollment"].values > 20,
    }
    subgroup_slices = {}
    for s_name, mask in slice_defs.items():
        if mask.sum() < 15 or len(np.unique(y_te[mask])) < 2:
            continue
        subgroup_slices[s_name] = {
            "rows": int(mask.sum()),
            "dropout_rate_pct": round(float(y_te[mask].mean() * 100), 1),
            "accuracy": round(float(accuracy_score(y_te[mask], best_te_pred[mask])), 4),
            "f1": round(float(f1_score(y_te[mask], best_te_pred[mask], zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_te[mask], best_te_probs[mask])), 4),
        }

    # --- 6. Export 5 Publication-Grade Figures ---
    plot_eda_and_class_balance(df_all, df_bin, FIGURES_DIR / "01_eda_and_class_balance.png")
    plot_training_curves(best_info["history"], best_info["best_epoch"], best_label, FIGURES_DIR / "02_training_curves.png")
    plot_roc_and_confusion(
        y_te,
        {
            "Perceptron (0 ocultas)": perc_te_scores,
            "Regressão Logística": lr_te_probs,
            "PyTorch MLP (selecionada)": best_te_probs,
        },
        np.array(perc_m["confusion_matrix"]),
        np.array(best_te_m["confusion_matrix"]),
        FIGURES_DIR / "03_roc_and_confusion.png",
    )
    plot_architecture_and_early_detection(arch_rows, early_rows, FIGURES_DIR / "04_architecture_and_early_detection.png")
    plot_importance_and_slices(imp_df, subgroup_slices, FIGURES_DIR / "05_feature_importance_and_slices.png")

    results = {
        "primary_task_binary_final_outcome": {
            "description": "Evasão (1) vs. Formado (0) — 3.630 estudantes com desfecho final conhecido",
            "splits": {
                "train": int(len(df_tr)),
                "val": int(len(df_va)),
                "test": int(len(df_te)),
                "encoded_input_dim": int(X_tr.shape[1]),
                "test_dropout_rate_pct": round(float(y_te.mean() * 100), 2),
            },
            "baselines": {
                "dummy_majority": dummy_m,
                "perceptron_linear": perc_m,
                "logistic_regression": lr_m,
            },
            "architecture_ablation": arch_rows,
            "selected_mlp": {
                "label": best_label,
                "hidden_layers": list(best_hidden),
                "num_trainable_layers": best_info["num_trainable_layers"],
                "input_dim_after_encoding": best_info["input_dim_after_encoding"],
                "num_parameters": best_info["num_parameters"],
                "best_epoch": best_info["best_epoch"],
                "val_auc_3seed_mean": best_arch_row["val_auc"],
                "val_threshold": round(best_thr, 2),
                "test_metrics": best_te_m,
            },
        },
        "early_detection_stages": early_rows,
        "all_students_binary_4424": all_students_binary,
        "multiclass_3_classes_4424": multiclass_3,
        "top15_permutation_importance": imp_df.head(15).round(5).to_dict(orient="records"),
        "subgroup_slices_test": subgroup_slices,
        "figures_generated": sorted(p.name for p in FIGURES_DIR.glob("*.png")),
    }

    out_path = REPORTS_DIR / "experiment_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"TAREFA PRINCIPAL: Evasao vs. Formado (n=3.630 | Entrada codificada={X_tr.shape[1]} dim)")
    print(f"Vencedora na Validacao (3 seeds): {best_label} ({best_info['num_parameters']:,} params)")
    print("=" * 78)
    print(f"  Dummy majoritario   | Acc={dummy_m['accuracy']:.4f} | F1=0.0000 | AUC=0.5000")
    print(f"  Perceptron (0 oc.)  | Acc={perc_m['accuracy']:.4f} | Prec={perc_m['precision']:.4f} | Rec={perc_m['recall']:.4f} | F1={perc_m['f1']:.4f} | AUC={perc_m['roc_auc']:.4f} {perc_m['auc_ci95']}")
    print(f"  Regressao Logistica | Acc={lr_m['accuracy']:.4f} | Prec={lr_m['precision']:.4f} | Rec={lr_m['recall']:.4f} | F1={lr_m['f1']:.4f} | AUC={lr_m['roc_auc']:.4f} {lr_m['auc_ci95']}")
    print(f"  PyTorch MLP selec.  | Acc={best_te_m['accuracy']:.4f} | Prec={best_te_m['precision']:.4f} | Rec={best_te_m['recall']:.4f} | F1={best_te_m['f1']:.4f} | AUC={best_te_m['roc_auc']:.4f} {best_te_m['auc_ci95']}")
    print("\nAblacao de arquiteturas:")
    for r in arch_rows:
        print(f"  {r['label']:26s} | params={r['num_parameters']:6,d} | Val AUC={r['val_auc']:.4f} | Test AUC={r['test_auc']:.4f} | Test Acc={r['test_acc']:.4f} | Test F1={r['test_f1']:.4f}")
    print("\nDeteccao precoce (PyTorch MLP):")
    for r in early_rows:
        print(f"  {r['stage_clean']:40s} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | AUC={r['roc_auc']:.4f}")
    print(f"\nTodos 4.424 alunos (Evasao vs Nao-Evasao) -> Perceptron AUC={all_students_binary['perceptron']['roc_auc']:.4f} | MLP AUC={all_students_binary['pytorch_mlp']['roc_auc']:.4f} (Acc={all_students_binary['pytorch_mlp']['accuracy']:.4f}, F1={all_students_binary['pytorch_mlp']['f1']:.4f})")
    print(f"Multiclasse 3 classes (4.424) -> Acc={multiclass_3['test_accuracy']:.4f} | Macro-F1={multiclass_3['test_macro_f1']:.4f} | Por classe={multiclass_3['f1_per_class']}")
    print("\nTop 8 variaveis por permutacao:")
    for r in imp_df.head(8).itertuples():
        print(f"  {r.label_pt:36s} [{r.group:24s}] | dAUC={r.delta_auc_mean:+.4f}")
    print(f"\nSalvos: {out_path} e {len(results['figures_generated'])} graficos em {FIGURES_DIR}")
    return results


if __name__ == "__main__":
    run_pipeline()
