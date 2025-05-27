# Laporan Proyek Machine Learning - Agistia Ronna Aniqa

![image](https://github.com/user-attachments/assets/2dd4587d-9d5f-495a-8f6f-d8ab6f00482e)

## Domain Proyek

Menurut **[Marlim et al (2022)](https://journal.ugm.ac.id/v3/JNTETI/article/view/3586)**, Diabetes merupakan penyakit metabolik kronis yang ditandai dengan peningkatan kadar glukosa darah yang lebih tinggi dari kadar normal, yang disebabkan oleh gangguan sekresi insulin atau gangguan efek biologis, atau keduanya. Diabetes menjadi salah satu penyakit kronis yang prevalensinya terus meningkat secara global. Penyakit ini dapat menyebabkan komplikasi serius di berbagai bagian tubuh dan secara keseluruhan meningkatkan risiko kematian dini. Beberapa komplikasi yang mungkin terjadi antara lain gagal ginjal, amputasi kaki, kehilangan penglihatan, serta kerusakan saraf. Selain itu, orang dewasa dengan diabetes memiliki risiko dua hingga tiga kali lipat lebih tinggi mengalami serangan jantung dan stroke. Pada masa kehamilan, diabetes yang tidak terkontrol dengan baik juga dapat meningkatkan risiko kematian janin dan komplikasi lainnya. Oleh karena itu, deteksi dini dan pengendalian penyakit diabetes sangat penting untuk mencegah dampak jangka panjang yang berbahaya.

Menurut penelitian yang dilakukan oleh **[Ginting et al. (2022)](https://jurnal.unprimdn.ac.id/index.php/jukep/article/view/2671)**, machine learning telah dimanfaatkan dalam deteksi dini dan prediksi risiko diabetes melitus tipe 2 dengan pendekatan scoping review terhadap 15 studi relevan dari tahun 2017–2021. Studi ini mengkaji pipeline analisis yang mencakup identifikasi faktor risiko, pemodelan prediktif, serta validasi performa model, menggunakan algoritma seperti Random Forest, Support Vector Machine, Decision Tree, Logistic Regression, dan Neural Networks. Beberapa studi juga mengombinasikan metode statistik dengan machine learning untuk mempersempit variabel penting. Penelitian ini menunjukkan bahwa algoritma Random Forest sering menjadi yang unggul dengan akurasi mencapai 97,88% dalam mendeteksi diabetes tahap awal. Variabel yang dipertimbangkan meliputi usia, jenis kelamin, indeks massa tubuh (BMI), riwayat keluarga, pola makan, kebiasaan merokok, aktivitas fisik, tekanan darah, dan faktor genetik, yang semuanya disesuaikan untuk mempermudah penerapan deteksi dini dalam praktik klinis serta mendukung pencegahan penyakit berbasis faktor risiko yang dapat diubah.

Selain itu, penelitian yang dilakukan oleh **[Ridwan et al. (2023)](https://teknokom.unwir.ac.id/index.php/teknokom/article/view/152)**  membahas penerapan algoritma machine learning seperti K-Nearest Neighbors, AdaBoost, Logistic Regression, Light Gradient Boosting, Random Forest, dan Support Vector Machine dalam mendeteksi diabetes mellitus. Studi ini menggunakan dataset Pima Indian dengan sembilan atribut kesehatan, termasuk jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, BMI, diabetes pedigree function, usia, dan outcome class. Untuk mengatasi masalah ketidakseimbangan kelas, penelitian ini menerapkan Synthetic Minority Over-sampling Technique (SMOTE) sehingga performa model meningkat. Hasil penelitian menunjukkan bahwa algoritma K-Nearest Neighbors mencapai akurasi tertinggi sebesar 82%, dengan precision, recall, dan F1-score yang seimbang, menunjukkan efektivitas model ini dalam meminimalkan kesalahan prediksi positif maupun negatif. Studi ini juga merekomendasikan eksplorasi lebih lanjut pada tuning parameter model dan penggunaan dataset yang lebih besar untuk meningkatkan generalisasi dan aplikasi klinis.

Berdasarkan penelitian tersebut, penerapan machine learning dalam prediksi diabetes merupakan solusi potensial untuk meningkatkan efisiensi deteksi dini sekaligus membantu pasien dalam pengambilan keputusan terkait gaya hidup dan perawatan medis. Deteksi dini sangat penting untuk menurunkan risiko komplikasi jangka panjang serta mengurangi biaya pengobatan yang tinggi. Dengan adanya model prediktif berbasis machine learning, tenaga medis dapat memberikan diagnosis yang lebih cepat dan akurat, sekaligus merekomendasikan tindakan pencegahan yang lebih tepat bagi pasien dengan risiko tinggi. Selain itu, pemanfaatan prediksi berbasis data dapat meningkatkan kesadaran masyarakat terhadap faktor-faktor risiko diabetes, sehingga mendorong perubahan gaya hidup ke arah yang lebih sehat. Oleh karena itu, implementasi model prediksi diabetes yang andal tidak hanya memberikan manfaat signifikan bagi individu, tetapi juga berkontribusi besar terhadap peningkatan kualitas layanan kesehatan secara menyeluruh.

## Business Understanding

### Problem Statements
1. Bagaimana cara memprediksi kemungkinan seseorang mengidap diabetes berdasarkan faktor-faktor kesehatan yang dimilikinya?
2. Seberapa akurat model machine learning dalam memprediksi diabetes dibandingkan dengan metode konvensional?

### Goals
1. Mengembangkan model machine learning untuk memprediksi kemungkinan seseorang mengidap diabetes berdasarkan faktor-faktor kesehatan yang relevan.
2. Mengevaluasi kinerja model machine learning menggunakan berbagai metrik evaluasi seperti akurasi, precision, recall, F1-score, dan confusion matrix guna memastikan performa optimal dalam mendeteksi diabetes.

### Solution Statement
1. Menerapkan beberapa algoritma machine learning seperti Random Forest, XGBoost, dan LightGBM untuk membandingkan performa model dalam mendeteksi diabetes.
2. Melakukan analisis terhadap hasil model berdasarkan metrik evaluasi, sehingga dapat dipilih model terbaik yang mampu memberikan prediksi dengan tingkat akurasi dan keandalan tertinggi.

## Data Understanding
**`Diabetes Prediction Dataset`** merupakan kumpulan data medis dan demografi pasien beserta status diabetes mereka (positif atau negatif). Dataset ini mencakup fitur-fitur seperti usia, jenis kelamin, indeks massa tubuh (BMI), hipertensi, penyakit jantung, riwayat merokok, kadar HbA1c, kadar glukosa darah, serta status diabetes. Dataset ini terdiri dari 9 kolom dan 100.000 baris, serta sudah bersih tanpa missing values. Dataset diambil dari platform **[Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)**.

### Variabel-variabel pada Diabetes Prediction Dataset adalah sebagai berikut:
- **`gender`** : Jenis kelamin biologis seseorang, yang dapat memengaruhi kerentanan terhadap diabetes. Terdapat tiga kategori: laki-laki, perempuan, dan lainnya.
- **`age`** : Usia merupakan faktor penting karena diabetes lebih umum terjadi pada orang dewasa yang lebih tua. Rentang usia pada dataset ini berkisar antara 0–80 tahun.
- **`hypertension`** : Hipertensi adalah kondisi tekanan darah tinggi. Nilai 0 menunjukkan tidak memiliki hipertensi, sedangkan nilai 1 menunjukkan memiliki hipertensi.
- **`heart_disease`** : Riwayat penyakit jantung. Nilai 0 menunjukkan tidak memiliki penyakit jantung, sedangkan nilai 1 menunjukkan memiliki penyakit jantung.
- **`smoking_history`** : Riwayat merokok sebagai salah satu faktor risiko diabetes dan dapat memperburuk komplikasi yang terkait dengan diabetes. Terdapat lima kategori: tidak saat ini, sebelumnya, tidak ada informasi, saat ini, tidak pernah, dan pernah.
- **`bmi`** : Indeks Massa Tubuh (BMI) adalah ukuran lemak tubuh berdasarkan berat dan tinggi badan. Nilai BMI yang tinggi dikaitkan dengan risiko diabetes yang lebih besar. Rentang BMI dalam dataset adalah 10,16–71,55. BMI <18,5 = kekurangan berat badan, 18,5–24,9 = normal, 25–29,9 = kelebihan berat badan, ≥30 = obesitas.
- **`HbA1c_level`** : Kadar HbA1c (Hemoglobin A1c) adalah ukuran kadar rata-rata gula darah selama 2–3 bulan terakhir. Nilai ≥6,5% biasanya menunjukkan diabetes.
- **`blood_glucose_level`** : Kadar glukosa darah pada waktu tertentu, yang merupakan indikator utama diabetes.
- **`diabetes`** : Variabel target, dengan nilai 1 menunjukkan pasien menderita diabetes, dan nilai 0 menunjukkan tidak.

### Visualisasi Distribusi Data Numerik
Visualisasi distribusi menunjukkan bahwa **age** memiliki sebaran yang relatif normal dengan sedikit skew di ujung kanan. **bmi** memiliki distribusi yang sangat tidak merata dengan puncak yang tajam di sekitar nol. **blood_glucose_level** memperlihatkan pola multimodal dengan beberapa puncak, sementara **HbA1c_level** menunjukkan pola serupa dengan banyak puncak dan distribusi yang tidak merata.

![DistribusiNumerik](https://github.com/user-attachments/assets/3d444c84-1f8f-4fac-b0b0-ffc1d37cfbf9)

### Visualisasi Distribusi Data Kategori
Visualisasi distribusi data kategori menunjukkan bahwa jumlah individu **perempuan** lebih banyak dibandingkan laki-laki. Mayoritas sampel **tidak memiliki hipertensi, penyakit jantung, maupun diabetes**, sementara hanya sebagian kecil yang terdiagnosis. Pada riwayat merokok, kategori "No Info" dan "Never" mendominasi, sedangkan kategori "Former", "Current", "Not Current", dan "Ever" memiliki jumlah yang jauh lebih sedikit.

![DistribusiKategori](https://github.com/user-attachments/assets/8872705e-99d6-4875-b7fa-e6a2624411df)

### Visualisasi Rata Rata Diabetes vs Fitur
Visualisasi menunjukkan bahwa rata-rata penderita diabetes lebih tinggi pada laki-laki dibanding perempuan. Individu dengan hipertensi dan penyakit jantung memiliki kemungkinan lebih besar menderita diabetes. Riwayat merokok juga berpengaruh, dengan mantan perokok (former smokers) memiliki rata-rata penderita diabetes tertinggi, sedangkan kategori "No Info" memiliki yang terendah.

![MeanDiabetesVSOtherFeatures](https://github.com/user-attachments/assets/bb9ce3b8-ebd9-447a-80a0-9e140a7816ec)

### Visualisasi KDE
Visualisasi menunjukkan hubungan antara berbagai fitur dalam dataset, seperti **age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level**, dan **diabetes**. Scatter plot memperlihatkan distribusi titik data di antara pasangan variabel, sementara plot KDE di diagonal menunjukkan distribusi probabilitas masing-masing variabel. Beberapa fitur seperti **HbA1c_level** dan **blood_glucose_level** memperlihatkan variasi distribusi yang jelas, sedangkan variabel biner seperti **hypertension** dan **heart_disease** menunjukkan pola distribusi yang lebih terpisah dan tidak terlalu kompleks.

![KDE](https://github.com/user-attachments/assets/7b102d56-64cb-4b21-b5d4-9dce80490a3c)

### Visualisasi Correlation Matrix
Visualisasi matriks korelasi menunjukkan hubungan antar fitur numerik. Terlihat bahwa **HbA1c_level** dan **blood_glucose_level** memiliki korelasi yang cukup tinggi dengan variabel target **diabetes** (masing-masing 0,38 dan 0,39), menandakan bahwa kadar gula darah dan HbA1c sangat penting dalam prediksi diabetes. Selain itu, terdapat korelasi moderat antara **bmi** dan **age (0,38)**, yang mungkin menunjukkan bahwa berat badan cenderung meningkat seiring bertambahnya usia. Namun, variabel seperti **heart_disease** dan **hypertension** memiliki korelasi lebih rendah dengan diabetes, yang menunjukkan faktor-faktor ini mungkin kurang dominan dibandingkan faktor lain dalam menentukan kondisi diabetes.

![CorrelationMatrix](https://github.com/user-attachments/assets/4f1cc1bb-96b7-4351-9a0f-73d4addad75f)

## Data Preparation
- **`Rare Category Handling`** <br>
  Pada fitur `gender`, terdapat kategori minoritas (misalnya kategori “other”) yang hanya muncul sebanyak 18 dari 100.000 baris. Oleh karena itu, kategori ini diganti dengan kategori mayoritas (male/female). Menggantikan kategori minoritas dengan modus membantu mengurangi noise dan ketidakseimbangan dalam fitur kategorikal, sehingga analisis dan pemodelan tidak terdistorsi oleh data yang terlalu jarang muncul. Hal ini memastikan model dapat belajar dari data yang lebih representatif.
  
     ```python
    gender_mode = df['gender'].mode()[0]

     df['gender'] = df['gender'].replace('Other', gender_mode)
     ```
- **`Handling Outlier`** <br>
  Outlier dicari pada kolom numerik menggunakan metode IQR (Interquartile Range). Setelah outlier terdeteksi, dilakukan clipping pada nilai yang berada di luar batas bawah dan atas yang ditentukan. Penanganan outlier penting karena nilai ekstrem dapat memengaruhi proses pelatihan model secara berlebihan. Dengan menerapkan clipping, data menjadi lebih bersih dan representatif, sehingga model dapat belajar pola secara optimal.
  
    ```python
    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    outlierValues = {}
    for col in numerical_features:
        q25 = np.percentile(df[col].dropna(), 25)
        q75 = np.percentile(df[col].dropna(), 75)
        iqr = q75 - q25
        lowerBound = q25 - 1.5 * iqr
        upperBound = q75 + 1.5 * iqr
        outliers = df[col][(df[col] < lowerBound) | (df[col] > upperBound)]
        outlierValues[col] = outliers
        df[col] = np.clip(df[col], lowerBound, upperBound)  
     ```
- **`Encoding Fitur Kategori`** <br>
  Fitur kategorikal seperti `gender` dan `smoking_history` dikonversi menjadi fitur numerik menggunakan teknik One-Hot Encoding. Encoding ini diperlukan karena sebagian besar algoritma machine learning hanya dapat memproses input numerik. Dengan konversi ini, informasi yang terkandung dalam kategori tetap terwakili tanpa membuat model salah menginterpretasikan urutan atau nilai antar kategori.
  
    ```python
    categorical_features = ['gender', 'smoking_history']
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_array = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_features), index=df.index)
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    ```
- **`Standarisasi`** <br>
  Fitur-fitur numerik seperti `age`, `bmi`, `blood_glucose_level`, dan `HbA1c_level` distandarisasi menggunakan StandardScaler, sehingga mayoritas nilai akan berada dalam rentang -3 hingga 3. Standarisasi ini penting untuk menyamakan skala antar fitur, membantu proses optimasi algoritma machine learning menjadi lebih stabil, cepat, serta mencegah bias akibat perbedaan skala yang signifikan.
  
    ```python
    numerical_features = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    ```
- **`Train-Test-Split`** <br>
  Dataset dibagi menjadi 80% data training dan 20% data testing, dengan diabetes sebagai variabel target yang akan diprediksi. Pemisahan ini penting untuk mengevaluasi kinerja model pada data yang belum pernah dilihat sebelumnya, membantu mengukur kemampuan generalisasi model, dan mencegah overfitting.

    ```python
    X = df.drop(columns=["diabetes"])  
    y = df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    ```

## Modeling
Pada studi ini digunakan tiga algoritma untuk memprediksi kemungkinan seseorang mengidap diabetes berdasarkan fitur-fitur kesehatan yang tersedia, yaitu **Random Forest**, **XGBoost**, dan **LightGBM**. Pemilihan ketiga model ini mempertimbangkan kekuatan mereka pada data tabular, serta kemampuan mereka menangani masalah klasifikasi yang kompleks.

### **Random Forest**

Random Forest adalah algoritma ensemble berbasis decision tree. Cara kerjanya:

* Membuat banyak decision tree dari **berbagai subset data acak (bootstrapping)**.
* Pada setiap split, hanya mempertimbangkan subset acak dari fitur (feature bagging).
* Untuk prediksi klasifikasi, hasil akhirnya didapat dengan **voting mayoritas** dari semua pohon (untuk regresi: rata-rata prediksi).

Kelebihan: tahan terhadap overfitting, mampu menangani data non-linear, robust terhadap outlier.

```python
model_randomforest = RandomForestClassifier(n_estimators=100, random_state=123)
model_randomforest.fit(X_train, y_train)
```

### **XGBoost**

XGBoost (Extreme Gradient Boosting) adalah algoritma boosting berbasis pohon keputusan dengan tambahan optimasi.
Cara kerjanya:

* **Membangun model secara sekuensial**, di mana setiap pohon baru dilatih untuk memperbaiki kesalahan (residual) dari pohon sebelumnya.
* Menggunakan teknik **gradient descent** untuk meminimalkan fungsi loss.
* Dilengkapi regularisasi (L1, L2) untuk mengurangi overfitting.

Kelebihan: performa tinggi pada data tabular, efektif untuk menangani kompleksitas pola, sering digunakan di kompetisi data science.

```python
model_xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123)
model_xgboost.fit(X_train, y_train)
```

### **LightGBM**

LightGBM adalah algoritma boosting mirip XGBoost, tetapi lebih dioptimalkan untuk efisiensi.
Cara kerjanya:

* **Gradient boosting**, seperti XGBoost, tetapi dengan teknik khusus:

  * **Leaf-wise tree growth** (lebih fokus memperdalam cabang dengan loss terbesar).
  * **Histogram-based splitting** untuk mempercepat proses.
* Cocok untuk dataset besar, tetapi lebih sensitif terhadap outlier.

Kelebihan: lebih cepat dan lebih hemat memori dibandingkan XGBoost, cocok untuk eksperimen skala besar.

```python
model_lightgbm = LGBMClassifier(random_state=123)
model_lightgbm.fit(X_train, y_train)
```

### Evaluasi Model

Setelah pelatihan, performa ketiga model dibandingkan menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score untuk menentukan model terbaik.

Berdasarkan hasil evaluasi awal, **LightGBM** dipilih sebagai model terbaik karena mampu memberikan hasil prediksi yang paling akurat dan seimbang dibandingkan dengan Random Forest dan XGBoost.

## Evaluation
**Evaluasi model** dilakukan menggunakan beberapa metrik utama yang sesuai dengan konteks klasifikasi biner, yaitu **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan **Confusion Matrix**. Metrik-metrik ini dipilih karena pada kasus prediksi diabetes, keseimbangan antara deteksi kasus positif dan negatif sangat penting untuk memastikan hasil yang akurat dan dapat diandalkan.

Metrik Evaluasi yang Digunakan
1. **`Accuracy Score`** :

    - **Accuracy** mengukur persentase prediksi yang benar dari seluruh prediksi.

        $ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $

        ```python
        test_acc = accuracy_score(y_test, y_test_pred)
        ```
    
2. **`Classification Report`** :
    - **Precision** mengukur proporsi prediksi positif yang benar.

        $ \text{Precision} = \frac{TP}{TP + FP} $

    - **Recall (Sensitivity)** mengukur proporsi kasus positif yang berhasil dideteksi.

        $ \text{Recall} = \frac{TP}{TP + FN} $

    - **F1-Score** adalah rata-rata harmonik antara Precision dan Recall, memberikan gambaran keseimbangan antara keduanya.  

        $ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $

        ```python
        print("\n--- Classification Report (Test) ---\n", classification_report(y_test, y_test_pred))
        ```

3. **`Confusion Matrix`** : 

    |                | Predicted Negatif (0) |  Predicted Positif (1) |
    |----------------|---------------|--------------------|
    | Actual Negatif (0)  | True Negative (TN)	        | False Positive (FP)              |
    | Actual Positif (1)        | False Negative (FN)	        | True Positive (TP)              |

     ```python
    test_cm = confusion_matrix(y_test, y_test_pred)
    ```

Berikut merupakan ringkasan hasil evaluasi berdasarkan prediksi pada data:
1. Accuracy dan Classification Report :

    | Model          | Accuracy |  Precision |  Recall |  F1-Score |
    |----------------|---------------|--------------------|-----------------|-------------------|
    | Random Forest  | 0.9683        | 0.93              | 0.68           | 0.79             |
    | XGBoost        | 0.9698        | 0.95              | 0.68           | 0.79             |
    | LightGBM       | 0.9705        | 0.97              | 0.67           | 0.80             |

    Analisis Hasil:
    - Ketiga model menunjukkan tingkat akurasi yang sangat tinggi (sekitar 97%), mengindikasikan kemampuan prediksi yang sangat baik pada data uji.
      
    - Precision yang tinggi (~0.95–0.97) menunjukkan bahwa model jarang salah mengklasifikasikan kasus negatif sebagai positif (False Positive).
    
      
    - Recall sedikit lebih rendah (~0.67–0.68), mengindikasikan masih adanya beberapa kasus positif yang tidak terdeteksi dengan baik (False Negative).

    - F1-Score menunjukkan XGBoost dan LightGBM memiliki keseimbangan yang cukup baik antara precision dan recall.

    Berdasarkan evaluasi ini, meskipun **LightGBM** memiliki akurasi tertinggi **(0.9705)**, perbedaannya dengan **XGBoost** dan **Random Forest** sangat kecil. Mengingat keseimbangan Precision dan Recall, XGBoost dipilih sebagai solusi final karena memberikan kombinasi terbaik dalam mendeteksi kasus positif tanpa banyak kesalahan klasifikasi.

3. Confusion Matrix :

    | Model         |    Actual           | Predicted Negatif (0) |  Predicted Positif (1)  |
    | -----         |----------------     |---------------        |--------------------     |
    |Random Forest  | Actual Negatif (0)  | 27310             	  | 128                     |
    |Random Forest  | Actual Positif (1)  | 822               	  | 1740                    |
    |XGBoost        | Actual Negatif (0)  | 27344             	  | 94                      |
    |XGBoost        | Actual Positif (1)  | 813               	  | 1749                    |
    |LightGBM       | Actual Negatif (0)  | 27389             	  | 49                      |
    |LightGBM       | Actual Positif (1)  | 835               	  | 1727                    |

    Analisis Confusion Matrix:
    - Ketiga model memiliki performa sangat baik dalam mengklasifikasikan kelas negatif (0), dengan jumlah **True Negative (TN)** yang tinggi dan **False Positive (FP)** yang sangat rendah, menunjukkan bahwa model jarang salah mengklasifikasikan kasus negatif sebagai positif.
    
    - Perbedaan muncul dalam menangani kelas positif (1), di mana **LightGBM memiliki jumlah False Negative (FN) tertinggi (835)**, artinya lebih banyak kasus positif yang tidak terdeteksi dibandingkan XGBoost (813) dan Random Forest (822).
    
    - **XGBoost menunjukkan keseimbangan terbaik** dengan False Negative yang lebih rendah dibandingkan LightGBM, dan False Positive yang lebih rendah dibandingkan Random Forest, menjadikannya model yang paling optimal untuk mendeteksi kasus positif secara akurat.

## Conclusion

Pada proyek ini, telah dikembangkan model machine learning untuk memprediksi kemungkinan seseorang mengidap diabetes berdasarkan berbagai faktor kesehatan, seperti usia, jenis kelamin, BMI, hipertensi, penyakit jantung, riwayat merokok, kadar HbA1c, dan kadar glukosa darah.

Tiga algoritma utama yang digunakan adalah **Random Forest**, **XGBoost**, dan **LightGBM**. Evaluasi dilakukan menggunakan metrik seperti **accuracy**, **precision**, **recall**, **F1-score**, dan **confusion matrix**. Ketiga model menunjukkan performa yang sangat baik, dengan tingkat akurasi sekitar **96.8%–97.0%**.

Meskipun **LightGBM** mencatatkan akurasi tertinggi (**0.9705**), evaluasi lebih lanjut menunjukkan bahwa **XGBoost** memiliki keseimbangan terbaik antara **precision (0.95)** dan **recall (0.68)**. Dengan jumlah **False Negative (813)** yang lebih rendah dibanding LightGBM (835), dan **False Positive (94)** yang masih tergolong rendah, XGBoost dinilai paling optimal dalam mendeteksi kasus positif secara akurat, tanpa mengorbankan banyak kesalahan klasifikasi negatif.

Kesimpulannya, penggunaan machine learning, khususnya dengan model **XGBoost**, terbukti efektif dalam meningkatkan efisiensi deteksi dini diabetes. Model ini berpotensi membantu tenaga medis dalam pengambilan keputusan klinis, sekaligus meningkatkan kesadaran masyarakat terhadap faktor risiko yang memicu penyakit. Dengan implementasi yang tepat, sistem prediksi ini dapat memberikan dampak positif tidak hanya bagi individu, tetapi juga bagi sistem kesehatan secara keseluruhan, melalui deteksi dan intervensi yang lebih cepat dan tepat.

## References

1. Marlim, Y. N., Suryati, L., & Agustina, N. (2022). Deteksi dini penyakit diabetes menggunakan machine learning dengan algoritma Logistic Regression. *Jurnal Nasional Teknik Elektro dan Teknologi Informasi*, 11(2), 88–96. [https://doi.org/10.22146/jnteti.v11i2.3586](https://doi.org/10.22146/jnteti.v11i2.3586)

2. Ginting, J., Ginting, R., & Hartono, H. (2022). Deteksi dan prediksi penyakit diabetes melitus tipe 2 menggunakan machine learning (scoping review). *Jurnal Keperawatan Priority*, 5(2), 93–105. [https://doi.org/10.34012/jukep.v5i2.2671](https://doi.org/10.34012/jukep.v5i2.2671)

3. Ridwan, A. M., & Setyawan, G. D. (2023). Perbandingan berbagai model machine learning untuk mendeteksi diabetes. *Teknokom*, 6(2), 127–132. [https://doi.org/10.31943/teknokom.v6i2.152](https://doi.org/10.31943/teknokom.v6i2.152)

