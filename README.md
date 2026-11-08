# CardioGuard AI — Predictive Sepsis & Hemodynamic Intelligence

**DataNova 2026 Global Data-Thon Submission · Healthcare & Social Impact Track**  
*Turning ICU Telemetry Into Proactive Bedside Decisions Hours Before Organ Collapse.*

---

## 1. Quick Deployment Guide (Zero-Config)

You can deploy **CardioGuard AI** in under 60 seconds using any of these three methods:

### Option A: Instant Static Hosting (Vercel / Netlify / GitHub Pages)
1. Upload this repository folder to GitHub or drag-and-drop it directly into [Netlify Drop](https://app.netlify.com/drop) or Vercel.
2. `index.html` at the root is a complete, zero-dependency clinical command center that runs immediately in the browser.

### Option B: Streamlit Community Cloud
1. Push this directory to a GitHub repository.
2. Connect the repository on [share.streamlit.io](https://share.streamlit.io) and select `streamlit_app.py` as the entrypoint.

### Option C: Run Locally
```bash
# Run the interactive web dashboard directly:
python3 -m http.server 8000
# Open http://localhost:8000 in your browser

# Or run the Python TCN + TreeSHAP inference engine:
python3 model/sepsis_tcn_engine.py
python3 fhir/fhir_stream_generator.py
```

---

## 2. Repository Structure
- `index.html` — Self-contained interactive ICU Bedside Command Center, Counterfactual Resuscitation Simulator, Multi-Center Validation Benchmarks, and HL7 FHIR R4 Bundle Inspector.
- `streamlit_app.py` — Streamlit Cloud wrapper for instant Python web hosting.
- `model/sepsis_tcn_engine.py` — Reference implementation of the 1D Kalman telemetry filter, dilated Temporal Convolutional Network (TCN) + BioClinicalBERT scoring pipeline, and TreeSHAP counterfactual simulator.
- `fhir/fhir_stream_generator.py` — HL7 FHIR Release 4 `Bundle`, `RiskAssessment`, and `Observation` payload generator.
- `docs/CardioGuard_AI_DataNova_2026_Pitch_Deck.pptx` — Official 11-slide Round 1 Proposal Pitch Deck.
- `assets/cardioguard_thumbnail_3x2.jpg` — Official 3:2 Devpost Project Thumbnail (<500 KB).
