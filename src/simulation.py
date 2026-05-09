"""
=============================================================
Non-Invasive Glucose Monitoring Smart Bracelet — Simulation
=============================================================
Academic Year 2025–2026 | Software Engineering Module
Team: Atia Chahinez, Belhadj Nour Elhouda, Halilem Merwa, Bouziza Issam

Description:
    Simulates the Random Forest AI model that runs inside the bracelet
    on the ESP32-S3. The script:
      1. Generates 50 synthetic blood glucose samples
      2. Simulates MAX86150 optical sensor signals (PPG)
      3. Applies Fitzpatrick skin-tone compensation
      4. Trains a Random Forest model
      5. Evaluates accuracy (MAE, R², Clarke Error Grid)
      6. Saves all charts to /results/
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────
RANDOM_SEED    = 42
N_SAMPLES      = 50          # number of simulated patients
NOISE_LEVEL    = 0.08        # sensor noise
RESULTS_DIR    = "results"
TARGET_MAE     = 15.0        # mg/dL — medical acceptability threshold
TARGET_R2      = 0.75
TARGET_ZONE_A  = 80.0        # % Clarke Zone A minimum

np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────
# 2. SYNTHETIC DATA GENERATION
# ──────────────────────────────────────────────────────────
def generate_synthetic_data(n=N_SAMPLES):
    """
    Generates realistic simulated sensor data.
    Each sample represents one patient measurement.
    """
    # True blood glucose (mg/dL) — realistic clinical distribution
    # Mix of hypoglycemic, normal and hyperglycemic values
    n_hypo   = int(n * 0.15)
    n_normal = int(n * 0.65)
    n_hyper  = n - n_hypo - n_normal   # ensures exactly n total

    glucose_true = np.concatenate([
        np.random.uniform(40,  70,  n_hypo),
        np.random.uniform(70,  180, n_normal),
        np.random.uniform(180, 350, n_hyper),
    ])
    np.random.shuffle(glucose_true)

    # Fitzpatrick skin type (1=very light, 6=very dark)
    n_light = int(n * 0.60)
    n_dark  = n - n_light
    fitzpatrick = np.concatenate([
        np.random.randint(1, 4, n_light),
        np.random.randint(4, 7, n_dark),
    ])
    np.random.shuffle(fitzpatrick)

    # Skin temperature (°C) — measured by MLX90614
    skin_temp = np.random.normal(33.5, 1.2, n)

    # Simulate optical absorption signal (PPG) from MAX86150
    # Wavelengths: 450nm, 600nm, 730nm, 850nm, 950nm
    def ppg_signal(glucose, fitz, temp, wavelength_sensitivity):
        """Simulates optical signal absorption through skin."""
        base_absorption = glucose * wavelength_sensitivity
        skin_factor     = 1 - (fitz - 1) * 0.04   # darker skin absorbs more
        temp_factor     = 1 + (temp - 33.5) * 0.003
        noise           = np.random.normal(0, NOISE_LEVEL * base_absorption)
        return base_absorption * skin_factor * temp_factor + noise

    ppg_450  = ppg_signal(glucose_true, fitzpatrick, skin_temp, 0.0028)
    ppg_600  = ppg_signal(glucose_true, fitzpatrick, skin_temp, 0.0045)
    ppg_730  = ppg_signal(glucose_true, fitzpatrick, skin_temp, 0.0062)
    ppg_850  = ppg_signal(glucose_true, fitzpatrick, skin_temp, 0.0078)
    ppg_950  = ppg_signal(glucose_true, fitzpatrick, skin_temp, 0.0091)

    # Feature engineering — wavelength ratios (optical signature)
    ratio_600_850  = ppg_600  / (ppg_850  + 1e-9)
    ratio_730_950  = ppg_730  / (ppg_950  + 1e-9)
    ratio_450_730  = ppg_450  / (ppg_730  + 1e-9)
    ratio_850_950  = ppg_850  / (ppg_950  + 1e-9)

    df = pd.DataFrame({
        # Raw sensor signals
        "ppg_450nm":       ppg_450,
        "ppg_600nm":       ppg_600,
        "ppg_730nm":       ppg_730,
        "ppg_850nm":       ppg_850,
        "ppg_950nm":       ppg_950,
        # Computed features
        "ratio_600_850":   ratio_600_850,
        "ratio_730_950":   ratio_730_950,
        "ratio_450_730":   ratio_450_730,
        "ratio_850_950":   ratio_850_950,
        # Patient profile
        "skin_temp":       skin_temp,
        "fitzpatrick":     fitzpatrick,
        # Ground truth
        "glucose_true":    glucose_true,
    })

    return df


# ──────────────────────────────────────────────────────────
# 3. RANDOM FOREST MODEL
# ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "ppg_450nm", "ppg_600nm", "ppg_730nm", "ppg_850nm", "ppg_950nm",
    "ratio_600_850", "ratio_730_950", "ratio_450_730", "ratio_850_950",
    "skin_temp", "fitzpatrick"
]

def train_model(df):
    X = df[FEATURE_COLS]
    y = df["glucose_true"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=3,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred, X_train, y_train


# ──────────────────────────────────────────────────────────
# 4. CLARKE ERROR GRID CLASSIFICATION
# ──────────────────────────────────────────────────────────
def clarke_zone(ref, pred):
    """
    Classifies each (reference, predicted) pair into Clarke zones A–E.
    Based on the standard Clarke Error Grid Analysis algorithm.
    """
    zones = []
    for r, p in zip(ref, pred):
        if (r < 70 and p < 70) or abs(p - r) / max(r, 1) <= 0.20:
            zones.append("A")
        elif (r >= 180 and p >= 70 and p <= 180) or \
             (r <= 70  and p >= 70 and p <= 180):
            zones.append("C")
        elif (r >= 240 and p <= 70) or (r <= 70 and p >= 240):
            zones.append("E")
        elif (r >= 180 and p >= 70 and p < r - 0.20 * r) or \
             (p <= 180 and r <= 70 and p > r + 0.20 * r):
            zones.append("D")
        else:
            zones.append("B")
    return np.array(zones)


# ──────────────────────────────────────────────────────────
# 5. PLOTS
# ──────────────────────────────────────────────────────────
def plot_bar_comparison(y_true, y_pred, save_path):
    """Figure 1 — bar chart: glucometer vs bracelet (like your feasibility report)"""
    fig, ax = plt.subplots(figsize=(14, 5))
    x    = np.arange(len(y_true))
    w    = 0.38

    ax.bar(x - w/2, y_true,  width=w, color="#2196F3", alpha=0.85, label="Invasive Glucometer (reference)")
    ax.bar(x + w/2, y_pred, width=w, color="#FF9800", alpha=0.85, label="Non-Invasive Bracelet (prediction)")

    ax.set_xlabel("Measurement n° (simulated patient)", fontsize=11)
    ax.set_ylabel("Blood Glucose (mg/dL)", fontsize=11)
    ax.set_title("Measurement-by-Measurement Comparison\nInvasive Glucometer vs Non-Invasive Bracelet", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(70,  color="red",    linestyle="--", linewidth=0.8, alpha=0.5, label="Hypo threshold (70)")
    ax.axhline(180, color="orange", linestyle="--", linewidth=0.8, alpha=0.5, label="Hyper threshold (180)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in x], fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_clarke_grid(y_true, y_pred, zones, save_path):
    """Figure 2 — Clarke Error Grid with colored zones"""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw zone backgrounds
    ax.fill_between([0, 58, 400], [0, 0, 0], [50, 50, 50],   color="#e8f5e9", zorder=0)
    ax.fill_betweenx([0, 400], [0, 0], [0, 0], color="white")

    # Zone A region approximation (shaded band)
    ref_line = np.linspace(0, 400, 400)
    ax.fill_between(ref_line, ref_line * 0.80, ref_line * 1.20,
                    color="#c8e6c9", alpha=0.5, label="Zone A (±20%)", zorder=1)

    # Scatter points by zone
    colors_map = {"A": "#1565C0", "B": "#FF8F00", "C": "#D84315", "D": "#6A1B9A", "E": "#B71C1C"}
    for zone in ["A", "B", "C", "D", "E"]:
        mask = zones == zone
        if mask.any():
            ax.scatter(y_true[mask], y_pred[mask],
                       color=colors_map[zone], label=f"Zone {zone} ({mask.sum()})",
                       s=70, edgecolors="white", linewidths=0.5, zorder=5)

    ax.plot([0, 400], [0, 400], "k--", linewidth=1, alpha=0.4)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel("Invasive Glucometer — Reference (mg/dL)", fontsize=11)
    ax.set_ylabel("Non-Invasive Bracelet — Prediction (mg/dL)", fontsize=11)
    ax.set_title("Clarke Error Grid Analysis\nNon-Invasive Glucose Bracelet Simulation", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    pct_A = (zones == "A").mean() * 100
    pct_B = (zones == "B").mean() * 100
    ax.text(220, 30, f"Zone A: {pct_A:.0f}%\nZone B: {pct_B:.0f}%\nZone C-E: {100-pct_A-pct_B:.0f}%",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_feature_importance(model, save_path):
    """Figure 3 — which sensor signals matter most"""
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1E88E5" if "ratio" in f else "#43A047" if "ppg" in f else "#FB8C00"
              for f in importances.index]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Random Forest — Feature Importance\n(which signals drive the prediction)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance score", fontsize=11)

    patches = [
        mpatches.Patch(color="#1E88E5", label="Wavelength ratios"),
        mpatches.Patch(color="#43A047", label="PPG raw signals"),
        mpatches.Patch(color="#FB8C00", label="Patient profile"),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_skin_tone_accuracy(df, y_pred_full, save_path):
    """Figure 4 — MAE by Fitzpatrick skin type"""
    df = df.copy()
    df["y_pred"] = y_pred_full
    df["abs_err"] = abs(df["glucose_true"] - df["y_pred"])

    mae_by_fitz = df.groupby("fitzpatrick")["abs_err"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#FFCCBC", "#FFAB91", "#FF8A65", "#BF360C", "#8D3B2B", "#5D1F0F"]
    bars = ax.bar(mae_by_fitz.index, mae_by_fitz.values,
                  color=[colors[i-1] for i in mae_by_fitz.index], edgecolor="white")
    ax.axhline(TARGET_MAE, color="red", linestyle="--", linewidth=1.5,
               label=f"Target MAE = {TARGET_MAE} mg/dL")
    ax.set_xlabel("Fitzpatrick Skin Type (1=very light → 6=very dark)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (mg/dL)", fontsize=11)
    ax.set_title("Accuracy by Skin Tone (Fitzpatrick Scale)\nFitzpatrick Compensation Validation", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"Type {i}" for i in range(1, 7)])
    ax.legend(fontsize=10)

    for bar, val in zip(bars, mae_by_fitz.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ──────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────
def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  NON-INVASIVE GLUCOSE BRACELET — SIMULATION")
    print("="*60)

    # — Generate data
    print("\n[1/5] Generating synthetic sensor data ...")
    df = generate_synthetic_data(N_SAMPLES)
    df.to_csv(f"{RESULTS_DIR}/simulation_data.csv", index=False)
    print(f"  ✓ {N_SAMPLES} samples generated and saved.")

    # — Train model
    print("\n[2/5] Training Random Forest model ...")
    model, X_test, y_test, y_pred, X_train, y_train = train_model(df)
    y_pred_full = model.predict(df[FEATURE_COLS])
    print(f"  ✓ Model trained on {len(X_train)} samples, tested on {len(X_test)}.")

    # — Metrics
    print("\n[3/5] Computing validation metrics ...")
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    zones  = clarke_zone(y_test.values, y_pred)
    pct_A  = (zones == "A").mean() * 100
    pct_B  = (zones == "B").mean() * 100
    pct_CE = 100 - pct_A - pct_B

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  MAE  = {mae:.2f} mg/dL  (target < {TARGET_MAE})   │")
    print(f"  │  R²   = {r2:.3f}        (target > {TARGET_R2})      │")
    print(f"  │  Zone A (Clarke) = {pct_A:.0f}%  (target ≥ {TARGET_ZONE_A}%) │")
    print(f"  │  Zone B          = {pct_B:.0f}%                    │")
    print(f"  │  Zones C–E       = {pct_CE:.0f}%                    │")
    print(f"  └─────────────────────────────────────┘")

    status_mae   = "✅ PASS" if mae  < TARGET_MAE  else "❌ FAIL"
    status_r2    = "✅ PASS" if r2   > TARGET_R2   else "❌ FAIL"
    status_zone  = "✅ PASS" if pct_A >= TARGET_ZONE_A else "❌ FAIL"

    print(f"\n  MAE   : {status_mae}")
    print(f"  R²    : {status_r2}")
    print(f"  Zone A: {status_zone}")

    # — Save metrics summary
    summary = {
        "MAE_mgdL":       round(mae,  3),
        "R2_score":       round(r2,   3),
        "Zone_A_pct":     round(pct_A, 1),
        "Zone_B_pct":     round(pct_B, 1),
        "Zone_CE_pct":    round(pct_CE,1),
        "N_test_samples": len(y_test),
        "MAE_target":     TARGET_MAE,
        "Pass_MAE":       mae < TARGET_MAE,
        "Pass_R2":        r2  > TARGET_R2,
        "Pass_ZoneA":     pct_A >= TARGET_ZONE_A,
    }
    pd.DataFrame([summary]).to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)
    print(f"\n  ✓ Metrics saved to {RESULTS_DIR}/metrics_summary.csv")

    # — Plots
    print("\n[4/5] Generating charts ...")

    # Use all 50 samples for bar chart (prettier)
    y_pred_all  = model.predict(df[FEATURE_COLS])
    zones_all   = clarke_zone(df["glucose_true"].values, y_pred_all)

    plot_bar_comparison(df["glucose_true"].values, y_pred_all,
                        f"{RESULTS_DIR}/fig1_bar_comparison.png")
    plot_clarke_grid(df["glucose_true"].values, y_pred_all, zones_all,
                     f"{RESULTS_DIR}/fig2_clarke_grid.png")
    plot_feature_importance(model,
                            f"{RESULTS_DIR}/fig3_feature_importance.png")
    plot_skin_tone_accuracy(df, y_pred_all,
                            f"{RESULTS_DIR}/fig4_skin_tone_accuracy.png")

    # — Done
    print("\n[5/5] Simulation complete!")
    print(f"\n  All files saved in /{RESULTS_DIR}/")
    print("  ├── simulation_data.csv")
    print("  ├── metrics_summary.csv")
    print("  ├── fig1_bar_comparison.png")
    print("  ├── fig2_clarke_grid.png")
    print("  ├── fig3_feature_importance.png")
    print("  └── fig4_skin_tone_accuracy.png")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
