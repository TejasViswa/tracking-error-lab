# 📈 Tracking Error Lab

**Interactive dashboard + technical write-up** exploring how *tracking error* differs when calculated on daily vs. monthly returns — complete with simulation, file uploads, and mathematical derivations.

---

## 🔗 Live Resources

- 🧮 **App (Streamlit):** [tracking-error-lab.streamlit.app](https://tracking-error-lab-kgkju98o4pqxdjoevqtsay.streamlit.app/)
- 📖 **Article (Quarto site):** [tejasviswa.github.io/tracking-error-lab](https://tejasviswa.github.io/tracking-error-lab/)
- 💻 **GitHub Repo:** [github.com/TejasViswa/tracking-error-lab](https://github.com/TejasViswa/tracking-error-lab)

---

## 🧠 Project Overview

This project demonstrates:

- The definition and interpretation of **tracking error (TE)** — how a portfolio’s active returns deviate from its benchmark.  
- The difference between **daily** and **monthly** tracking error measurements.  
- How autocorrelation in returns can affect TE scaling.  
- An **interactive Streamlit simulator** for experimenting with AR(1) active-return processes.  
- A **file-upload tool** to compute realized TE from your own portfolio and benchmark data.

---

## 🗂️ Repository Structure

| Folder | Description |
|:-------|:-------------|
| **`/app`** | Streamlit dashboard (simulation, CSV upload, TE calculation) |
| **`/site`** | Quarto write-up (math, examples, documentation) |
| **`.github/workflows/`** | GitHub Action for automatic Quarto site publishing |
| **`.streamlit/`** | Streamlit theming configuration |

---

## ⚙️ Local Setup

Clone and run locally:

```bash
git clone https://github.com/TejasViswa/tracking-error-lab.git
cd tracking-error-lab/app
pip install -r requirements.txt
streamlit run app.py
