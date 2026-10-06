Klasifikasi Hasil Clustering Video Game Sales

## Project Overview

Dalam industri video game, memahami segmentasi produk berdasarkan performa penjualan dan karakteristik game sangat penting untuk strategi pemasaran. Proyek ini merupakan kelanjutan dari proses clustering (pengelompokan) data penjualan video game. Tujuan utama dari proyek ini adalah membangun model klasifikasi Machine Learning yang mampu memprediksi kategori (cluster) dari sebuah video game berdasarkan fitur-fitur seperti platform, genre, dan angka penjualan di berbagai wilayah. 

## Business Understanding

### Problem Statements
- Bagaimana cara membangun model klasifikasi yang akurat untuk memprediksi cluster dari sebuah video game berdasarkan data penjualannya?
- Algoritma mana yang memberikan performa klasifikasi terbaik antara K-Nearest Neighbors (KNN) dan Decision Tree untuk dataset ini?

### Goals
- Mengembangkan model machine learning klasifikasi untuk memprediksi target `Cluster`.
- Membandingkan performa model K-Nearest Neighbors (KNN) dan Decision Tree, baik sebelum maupun sesudah dilakukan proses hyperparameter tuning.

### Solution Approach
- **K-Nearest Neighbors (KNN)**: Algoritma berbasis jarak yang membandingkan sampel baru dengan sampel terdekat di dataset latih.
- **Decision Tree**: Algoritma berbasis pohon keputusan yang memecah data berdasarkan aturan kondisional dari fitur-fitur yang ada.
- Melakukan optimasi model menggunakan teknik **GridSearchCV** untuk mencari kombinasi hyperparameter terbaik.

## Data Understanding

Dataset yang digunakan adalah `vgsales_clustered.csv` yang berisi rincian penjualan video game global beserta label `Cluster` sebagai variabel target.

Fitur-fitur utama dalam dataset meliputi:
- `Rank`, `Name`: Informasi dasar video game.
- `Platform`, `Genre`: Fitur kategorikal yang mendeskripsikan jenis dan konsol game.
- `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`, `Total Sales`: Fitur numerik representasi angka penjualan.
- `Cluster`: Label klasifikasi target (bernilai 0 atau 1).

## Data Preparation

Tahapan persiapan data yang dilakukan sebelum pemodelan meliputi:
- **Pembersihan Data**: Menghapus fitur yang tidak relevan untuk prediksi, yaitu `Rank` dan `Name`.
- **Encoding Kategorikal**: Menerapkan `LabelEncoder` pada kolom kategorikal (`Platform` dan `Genre`) agar dapat diproses oleh algoritma.
- **Pembagian Dataset**: Memisahkan data menjadi data latih (70%) dan data uji (30%) menggunakan `train_test_split`.
- **Standarisasi Fitur**: Menggunakan `StandardScaler` untuk menormalisasi skala data numerik, yang sangat krusial untuk algoritma berbasis jarak seperti KNN.

## Modeling

### 1. K-Nearest Neighbors (KNN)
- **Base Model**: Dilatih menggunakan parameter default (`n_neighbors=5`).
- **Tuned Model**: Setelah GridSearchCV, kombinasi hyperparameter terbaik yang ditemukan adalah `metric: euclidean`, `n_neighbors: 7`, dan `weights: uniform`.

### 2. Decision Tree Classifier
- **Base Model**: Dilatih menggunakan parameter default.
- **Tuned Model**: Hasil GridSearchCV menemukan parameter terbaik berupa `criterion: entropy`, `max_depth: None`, `min_samples_leaf: 4`, dan `min_samples_split: 2`.

## Evaluation

Model dievaluasi menggunakan metrik **Accuracy**, **F1-Score**, **Precision**, dan **Recall**, serta visualisasi menggunakan **Confusion Matrix**.

Hasil evaluasi menunjukkan performa yang hampir sempurna pada data uji:
- **KNN (Default)**: Accuracy 0.9996, F1-Score 0.9996
- **Decision Tree (Default)**: Accuracy 0.9994, F1-Score 0.9994
- **KNN (Tuned)**: Accuracy 0.9993, F1-Score 0.9996
- **Decision Tree (Tuned)**: Accuracy 0.9995, F1-Score 0.9997

**Kesimpulan Evaluasi:**
Kedua model menunjukkan performa prediksi yang sangat tinggi. Proses tuning berhasil mempertahankan tingkat akurasi di atas 99%. Namun, nilai metrik yang mendekati 1.0 ini mengindikasikan bahwa pola pada dataset sangat mudah dipisahkan, atau terdapat potensi overfitting. Tindakan lanjutan seperti K-Fold Cross Validation dan Feature Importance direkomendasikan untuk analisis lebih mendalam di masa mendatang.

## Setup Environment & Cara Menjalankan

### Persiapan Direktori dan Dependensi
Pastikan Anda memiliki Python 3 terinstal, kemudian jalankan perintah berikut di terminal:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Menjalankan Notebook
1. Buka terminal atau command prompt.
2. Navigasikan ke direktori tempat file proyek disimpan.
3. Jalankan Jupyter Notebook dengan perintah:
```bash
jupyter notebook
```
4. Buka file `[Klasifikasi]_Submission_Akhir_BMLP_Wiraldy_Manggala_Revisi.ipynb` di peramban web dan jalankan seluruh sel (Run All).

## Kontributor
- Wiraldy Manggala Simanjuntak
