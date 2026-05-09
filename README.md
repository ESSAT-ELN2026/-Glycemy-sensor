# 🩺 Non-Invasive Glucose Monitoring Smart Bracelet
### AI Simulation — Random Forest Model | Software Engineering Module 2025–2026

---

## 👥 Team

| Name | Role |
|---|---|
| Atia Chahinez | Project Lead & AI Developer |
| Belhadj Nour Elhouda Khadidja | Firmware & Hardware |
| Halilem Merwa | Signal Processing |
| Bouziza Issam | Mobile App & Connectivity |

  

---

## 📌 Project Overview

More than **537 million people** worldwide have diabetes. In Algeria alone, **1.5 to 2 million patients** must prick their finger 3–5 times per day — painful, expensive (~5,000 DZD/month), and non-continuous.

This project proposes a **connected smart bracelet** that measures blood glucose **without any needle**, using:

- **MAX86150** optical sensor (PPG) on the wrist
- **ESP32-C3** microcontroller with embedded AI
- **MLX90614** infrared thermometer for skin correction
- **Random Forest** model (TensorFlow Lite Micro)
- **Bluetooth 5.0** mobile app

This repository contains the **Python simulation** of the AI algorithm — the core intelligence running inside the bracelet.

---

## 📊 Simulation Results

| Metric | Result | Target | Status |
|---|---|---|---|
| Mean Absolute Error (MAE) | **12.99 mg/dL** | < 15 mg/dL | ✅ PASS |
| Correlation (R²) | **0.922** | > 0.75 | ✅ PASS |
| Clarke Error Grid — Zone A | **93%** | ≥ 80% | ✅ PASS |
| Clarke Error Grid — Zone B | 7% | — | ✅ |
| Dangerous Zones (C–E) | **0%** | 0% | ✅ PASS |

> ⚠️ These results are based on **simulated data**. Real-patient validation (Phase 3) will be conducted at **CHU Tlemcen** on 50 volunteers (≥40% Fitzpatrick types 4–6).

---

## 🗂️ Repository Structure

```
glucose-bracelet/
│
├── src/
│   └── simulation.py          # Main simulation script
│
├── results/                   # Auto-generated when you run the script
│   ├── simulation_data.csv    # 50 synthetic patient samples
│   ├── metrics_summary.csv    # MAE, R², Clarke zones
│   ├── fig1_bar_comparison.png
│   ├── fig2_clarke_grid.png
│   ├── fig3_feature_importance.png
│   └── fig4_skin_tone_accuracy.png
│
├── docs/
│   ├── Feasibility_Report.pdf
│   ├── Planification_Report.pdf
│   └── UserStory_UML_Report.pdf
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/glucose-bracelet.git
cd glucose-bracelet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the simulation
```bash
python src/simulation.py
```

That's it! The script will:
1. Generate 50 synthetic patient samples
2. Train the Random Forest model
3. Evaluate accuracy (MAE, R², Clarke Grid)
4. Save 4 charts in `/results/`

---

## 🧠 How the Algorithm Works

The simulation mirrors the exact logic embedded in the bracelet firmware:

```
Patient wears bracelet
        ↓
MAX86150 emits light (450–950 nm) through wrist skin
        ↓
MLX90614 measures skin temperature
        ↓
ESP32-S3: filter noise + Fitzpatrick skin compensation
        ↓
Extract wavelength ratios (optical signature)
        ↓
Random Forest predicts glucose in mg/dL
        ↓
OLED display + LED alert + Bluetooth → Mobile App
```

### Key features used by the model:
- **5 PPG wavelength signals** (450, 600, 730, 850, 950 nm)
- **4 wavelength ratios** (optical glucose signature)
- **Skin temperature** (MLX90614 compensation)
- **Fitzpatrick type** (skin tone correction, types 1–6)

---

## 🎯 Validation — Clarke Error Grid

The Clarke Error Grid is the standard medical tool for evaluating glucometer accuracy:

| Zone | Meaning | Our Result |
|---|---|---|
| **A** | Clinically acceptable (< 20% error) | **93%** ✅ |
| **B** | Acceptable with caution (20–40% error) | 7% |
| **C–E** | Dangerous — incorrect treatment risk | **0%** ✅ |

---

## 🌍 Skin Tone Fairness (Fitzpatrick Scale)

A key innovation of this project is **automatic compensation for skin tone** using the Fitzpatrick scale (1–6). At least **40% of test volunteers** will have dark skin (types 4–6) to validate that the algorithm is equally accurate for all patients — addressing a known weakness of existing non-invasive solutions.

---

## 📅 Project Timeline

| Quarter | Phase | Key Deliverable |
|---|---|---|
| Q1 (M1–3) | Research & Algorithm | ← **This simulation** |
| Q2 (M4–6) | Prototype Development | PCB + Firmware v1 + Bracelet |
| Q3 (M7–9) | Volunteer Testing | CHU Tlemcen — 50 patients |
| Q4 (M10–12) | Finalization & Regulatory | ANPP file + Final Report |

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is developed for academic purposes at the University of Tlemcen.  
Academic Year 2025–2026 — Software Engineering Module.

---

## 📬 Contact

For any questions about this project, please contact the team via the university module supervisor.
