# Data Science Intern Case Study — Requirements

## Overview
- Dataset: Physical Medicine & Rehabilitation
- Size: **2235 observations**, **13 features**
- Target: **TedaviSuresi** (treatment duration in sessions)
- Goal: Conduct **in-depth EDA** and make the data **model-ready** (clean, consistent, analyzable). **No modeling required.**

## Columns of Dataset
- **HastaNo** — Anonymized patient ID
- **Yas** — Age
- **Cinsiyet** — Gender
- **KanGrubu** — Blood type
- **Uyruk** — Nationality
- **KronikHastalik** — Chronic conditions (comma-separated list)
- **Bolum** — Department/Clinic
- **Alerji** — Allergies (may be single or comma-separated)
- **Tanilar** — Diagnoses
- **TedaviAdi** — Treatment name
- **TedaviSuresi** — Treatment duration in sessions (**target**)
- **UygulamaYerleri** — Application sites
- **UygulamaSuresi** — Application duration

## Tasks
1) **Exploratory Data Analysis (EDA)**
   - Use Python with **Pandas, Matplotlib, Seaborn** to understand structure and variable types
   - Detect anomalies and missing data
   - Visualize (histograms, scatter plots, heatmaps) to uncover patterns and relationships
2) **Data Pre-Processing**
   - Handle missing values (e.g., **SimpleImputer**, **KNNImputer**)
   - Encode categorical variables (e.g., **OneHotEncoder**, **LabelEncoder**)
   - Normalize/standardize numerical features as needed
   - Address overall data quality issues

## Optional (Nice to Have)
- **Documentation:** Prepare a document summarizing EDA findings and preprocessing steps. Include it in the repo. **Put your name, surname, and email at the top.**
- **Pipeline Level Code:** Organize code at the pipeline level; ensure modularity and reusability.
- **Different Approach:** Alternative approaches are welcome.

## Submission Requirements
- **GitHub Repository**
  - Include code and a **README.md** with an overview, how to run, and other relevant info. **Put your name, surname, and email at the top of the README.**
  - If you prepared a findings document, include it too (**with name, surname, email at the top**).
  - Everyone must create their own repository and send the GitHub link to the provided email when complete.
  - Repo name format: **Pusula_Name_Surname** (for this project: **Pusula_Beyza_Ozbay**).
- **NOTE:** Those who do not comply with the Submission Requirements will not be considered for evaluation.
- **Deadline:** **06.09.2025 23:59**

## Compliance Checklist
- [ ] Data file placed at `data/raw/Talent_Academy_Case_DT_2025.xlsx`
- [ ] 2235 rows & 13 columns verified; **TedaviSuresi** present
- [ ] EDA performed with visuals (hist, scatter, heatmap)
- [ ] Preprocessing: missing values, encoding, scaling (if needed), data quality fixes
- [ ] Clean/model-ready data saved under `data/processed/`
- [ ] **README.md** includes name, surname, email + run instructions
- [ ] (Optional) Findings document in repo (includes name, surname, email)
- [ ] Repo name: **Pusula_Beyza_Ozbay**
- [ ] Submitted before **06.09.2025 23:59**
