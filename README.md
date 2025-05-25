# 🔍 Predictive Analysis – Proyek Machine Learning Terapan

Proyek ini bertujuan untuk membangun model **machine learning** yang mampu memprediksi kemungkinan seseorang mengidap diabetes berdasarkan dataset yang tersedia. Model yang digunakan mencakup **Random Forest**, **XGBoost**, dan **LightGBM**, dengan evaluasi menggunakan metrik seperti **accuracy**, **precision**, **recall**, dan **F1-score**.

---

## 🗂️ Struktur Proyek

* 📓 **Predictive\_Analysis.ipynb** → Notebook eksplorasi data, preprocessing, pemodelan, dan evaluasi.
* 📝 **Laporan\_Predictive\_Analysis.md** → Laporan lengkap berisi ringkasan alur proyek dan hasil analisis.
* 📊 **diabetes\_prediction\_dataset.csv** → Dataset utama yang digunakan dalam proyek.
* 📦 **requirements.txt** → Daftar dependensi/libraries yang diperlukan untuk menjalankan proyek.

---

## ⚙️ Setup Environment

### 🐍 **Menggunakan Anaconda**

```bash
conda create --name predictive-analysis python=3.9
conda activate predictive-analysis
pip install -r requirements.txt
```

### 💻 **Menggunakan Pipenv**

```bash
pipenv install
pipenv shell
pip install -r requirements.txt
```

---

## 🎯 Tujuan Proyek

1. 🔬 Membandingkan performa beberapa model machine learning untuk prediksi diabetes.
2. 📈 Melakukan analisis mendalam terhadap hasil model untuk menentukan model yang paling optimal.
3. 🌐 Menyediakan dashboard atau visualisasi interaktif untuk membantu pemahaman hasil prediksi dan mendukung pengambilan keputusan.
