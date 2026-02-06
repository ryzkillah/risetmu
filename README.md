📘 README.md — risetmu
# RisetMU  
**Intrusion Detection System (IDS) with Explainable AI (XAI)**

Repository ini berisi **kode, eksperimen, dan hasil analisis** untuk penelitian
Intrusion Detection System (IDS) berbasis Machine Learning yang dilengkapi dengan
Explainable Artificial Intelligence (XAI).

Repo ini **tidak menyertakan dataset mentah atau dataset berukuran besar**,
sesuai praktik terbaik repositori riset dan batasan GitHub.

---

## 📂 Struktur Repository



risetmu/
├── data/
│ └── sample/ # (opsional) dataset kecil untuk demo / reproducibility
│
├── modelling/ # Modul training & evaluasi model
│
├── scripts/ # Script utilitas (preprocessing, helper, dsb.)
│
├── xai/ # Modul Explainable AI (SHAP, LIME, dsb.)
│
├── results/
│ ├── figures/ # Visualisasi hasil (confusion matrix, SHAP, dll.)
│ └── metrics/ # Ringkasan metrik & metadata eksperimen
│
├── paper/ # Draft dan material paper ilmiah
│
├── modeling_baseline.py # Baseline model IDS
├── gabungkan.py # Script penggabungan dataset
├── merge_unsw_nb15.py # Preprocessing UNSW-NB15
├── xai_integration.py # Integrasi XAI ke pipeline IDS
│
├── .gitignore
├── .gitattributes
└── README.md


---

## 🧪 Dataset

### 🔴 Raw & Full Dataset
Dataset mentah dan dataset penuh **tidak disertakan di repository ini**.

Sumber resmi:
- **CICIDS2017**  
  https://www.unb.ca/cic/datasets/ids-2017.html
- **UNSW-NB15**  
  https://research.unsw.edu.au/projects/unsw-nb15-dataset

Dataset mentah dan dataset penuh **diarsipkan secara terpisah** di Zenodo 10.5281/zenodo.18509357
untuk keperluan replikasi dan sitasi ilmiah.

---

### 🟡 Sample Dataset (Opsional)
Repository ini dapat menyertakan **dataset sample berukuran kecil** untuk:
- demonstrasi pipeline,
- pengujian cepat,
- reproduktibilitas dasar.

Lokasi:


data/sample/


---

## ⚙️ Alur Eksperimen Singkat

1. Preprocessing dataset (external / sample)
2. Training model IDS (baseline & komparatif)
3. Evaluasi performa (accuracy, precision, recall, F1, confusion matrix)
4. Analisis XAI (SHAP, LIME)
5. Visualisasi & interpretasi hasil

---

## 📊 Hasil & Visualisasi
Hasil eksperimen tersimpan di:


results/


Termasuk:
- Confusion matrix
- Feature importance
- SHAP summary & bar plot
- LIME explanation

---

## 📄 Paper
Folder `paper/` digunakan untuk:
- draft artikel ilmiah,
- tabel & gambar final,
- catatan revisi reviewer.

---

## 🔁 Reproducibility
Untuk mereplikasi eksperimen penuh:
1. Unduh dataset dari sumber resmi / arsip eksternal
2. Letakkan dataset sesuai struktur yang dijelaskan pada dokumentasi
3. Jalankan script preprocessing dan training

---

## 📌 Catatan Penting
- GitHub **bukan tempat penyimpanan dataset besar**
- Dataset mentah disediakan melalui repositori data khusus (mis. Zenodo)
- Repo ini difokuskan pada **kode, metodologi, dan hasil**

---

## 📜 Lisensi
Kode dalam repository ini digunakan untuk **kepentingan akademik dan riset**.
Lisensi dataset mengikuti **ketentuan dari penyedia dataset asli**.

---

## 👤 Author
**RisetMU Team**  
Universitas Muhammadiyah Mataram
