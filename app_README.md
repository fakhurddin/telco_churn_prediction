# 🖥️ Streamlit App — `app.py`

The interactive web interface for the Customer Churn Predictor. Enter any customer's details and get an instant prediction with probability, risk badge, SHAP explanations, and retention recommendations.

---

## ✅ Status: COMPLETE

---

## 🚀 How to Run

```bash
# From the project root:
streamlit run app.py
```

Opens at: **http://localhost:8501**

> ⚠️ Run all `src/` pipeline scripts first — the app needs the trained model files.

---

## 📋 Prerequisites Checklist

```
models/
├── best_model.pkl       ✅  (run train_model.py)
├── scaler.pkl           ✅  (run feature_engineering.py)
├── encoder.pkl          ✅  (run feature_engineering.py)
├── feature_cols.pkl     ✅  (run feature_engineering.py)
└── shap_explainer.pkl   ✅  (run train_model.py)
```

If any are missing, run the full pipeline:
```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/train_model.py
```

---

## 🎛️ UI Layout — Full Breakdown

### Sidebar (Dark Navy Panel)

Divided into 4 sections:

#### 📋 Account Info
| Widget | Type | Feature | Range |
|--------|------|---------|-------|
| Tenure | Slider | `tenure` | 0–72 months |
| Monthly Charges | Number input | `MonthlyCharges` | $18–$120 |
| Total Charges | Auto-calculated | `TotalCharges` | tenure × monthly |
| Contract Type | Dropdown | `Contract` | Month-to-month / One year / Two year |
| Payment Method | Dropdown | `PaymentMethod` | 4 options |
| Paperless Billing | Checkbox | `PaperlessBilling` | Yes / No |

#### 🌐 Internet & Services
| Widget | Type | Feature |
|--------|------|---------|
| Internet Service | Dropdown | `InternetService` |
| Online Security | Radio | `OnlineSecurity` |
| Online Backup | Radio | `OnlineBackup` |
| Device Protection | Radio | `DeviceProtection` |
| Tech Support | Radio | `TechSupport` |
| Streaming TV | Radio | `StreamingTV` |
| Streaming Movies | Radio | `StreamingMovies` |

> If Internet Service = "No", all add-on fields auto-set to "No internet service"

#### 📱 Phone
| Widget | Type | Feature |
|--------|------|---------|
| Phone Service | Radio | `PhoneService` |
| Multiple Lines | Radio | `MultipleLines` |

> If Phone Service = "No", Multiple Lines auto-sets to "No phone service"

#### 👤 Demographics
| Widget | Type | Feature |
|--------|------|---------|
| Gender | Radio | `gender` |
| Senior Citizen | Checkbox | `SeniorCitizen` |
| Partner | Radio | `Partner` |
| Dependents | Radio | `Dependents` |

---

### Main Panel — 5 Result Sections

#### Section 1 — Churn Probability (top left)
- Large **probability number** in risk colour (e.g. `73.4%`)
- Animated **semicircular gauge** with colour zones and moving needle

#### Section 2 — Risk Level (top centre)
- Colour-coded **risk badge**: 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH
- Plain-English status message
- **Progress bar breakdown** — will stay % vs will churn %

#### Section 3 — Customer Profile (top right)
- 8-row quick-reference card with key input values

#### Section 4 — SHAP Waterfall Chart (bottom left)
- Top 5 feature drivers — red = toward churn, green = reduces risk

#### Section 5 — Retention Recommendations (bottom right)
- Up to 3 actionable retention actions
- Urgency banner based on risk level

---

## 🔧 Internal Flow

```
User clicks "Predict Churn Risk"
          │
          ▼
app.py assembles 19-feature customer dict from sidebar
          │
          ▼
predict.preprocess_input()
  ├── scaler.transform()    (scale numeric cols)
  └── encoder.transform()   (one-hot encode 15 categorical cols)
          │
          ▼
best_model.predict_proba()  →  churn probability
          │
          ▼
shap_explainer.shap_values()  →  top 5 drivers
          │
          ▼
get_recommendations()  →  retention actions
          │
          ▼
Streamlit renders all 5 sections
```

---

## 🎨 Design

| Element | Colour |
|---------|--------|
| Sidebar | `#1E2A3A` dark navy |
| LOW risk | `#1D9E75` teal green |
| MEDIUM risk | `#E8A838` amber |
| HIGH risk | `#E24B4A` red |
| Cards | `#FFFFFF` white |
| App bg | `#F4F6F9` light grey |

---

## 🐞 Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: models/best_model.pkl` | Run pipeline scripts first |
| `ModuleNotFoundError: streamlit` | `pip install -r requirements.txt` |
| `ValueError: feature names mismatch` | Re-run `feature_engineering.py` + `train_model.py` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| SHAP chart missing | Re-run `train_model.py` |

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push project to a public GitHub repo (include `.pkl` files)
2. Go to https://share.streamlit.io → New app
3. Set main file: `app.py`
4. Click Deploy

---

## ✅ Complete Feature Checklist

| Feature | Status |
|---------|--------|
| 18 sidebar inputs (all 4 sections) | ✅ |
| Auto-calculated TotalCharges | ✅ |
| Internet/phone conditional inputs | ✅ |
| Semicircular probability gauge | ✅ |
| Risk badge — LOW / MEDIUM / HIGH | ✅ |
| Progress bar breakdown | ✅ |
| Customer profile card | ✅ |
| SHAP waterfall chart | ✅ |
| Retention recommendations | ✅ |
| Urgency banner by risk level | ✅ |
| Model info footer | ✅ |
| Error state for missing model | ✅ |

---

*Phase 7 complete — full project built.*
