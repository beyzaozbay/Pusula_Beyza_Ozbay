# Data Dictionary

| Column            | Type*        | Description                                                | Notes                                                     |
|-------------------|--------------|------------------------------------------------------------|-----------------------------------------------------------|
| HastaNo           | identifier   | Anonymized patient ID                                      | Expected unique; check duplicates                         |
| Yas               | numeric      | Age                                                        | Check range/outliers; cast to integer if needed           |
| Cinsiyet          | categorical  | Gender                                                     | Normalize labels/case                                     |
| KanGrubu          | categorical  | Blood type                                                 | A, B, AB, 0 (+/-); unify notation                         |
| Uyruk             | categorical  | Nationality                                                | Normalize spelling/case                                   |
| KronikHastalik    | multi-label  | Chronic conditions (comma-separated)                       | Split by comma; trim; multi-hot encode                    |
| Bolum             | categorical  | Department/Clinic                                          | Standardize names                                         |
| Alerji            | multi-label  | Allergies (may be single or comma-separated)               | Split by comma; multi-hot encode                          |
| Tanilar           | text         | Diagnoses                                                  | Free text; optional parsing/standardization               |
| TedaviAdi         | categorical  | Treatment name                                             | Consider grouping rare categories                         |
| TedaviSuresi      | numeric      | Treatment duration in sessions (**target**)                | Non-negative; distribution check                          |
| UygulamaYerleri   | multi-label  | Application sites                                          | Split by comma; multi-hot encode                          |
| UygulamaSuresi    | numeric      | Application duration                                       | Units consistency; potential scaling                      |

\* Types are provisional; confirm via EDA and adjust preprocessing accordingly.

## Acceptance Checks
- [ ] Exactly **13** columns present with the exact names above
- [ ] **2235** rows in the dataset
- [ ] Target column **TedaviSuresi** exists and is numeric or castable
- [ ] Multi-label fields parsed consistently (comma-separated, trimmed)
