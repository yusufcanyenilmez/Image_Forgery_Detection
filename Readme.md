# 🛡️ Görüntü Sahteciliği Tespit Sistemi (Image Forgery Detection)

<br>

<div align="center">
<img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" />
<img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react" />
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv" />
<img src="https://img.shields.io/badge/PyTorch-AI-orange?style=for-the-badge&logo=pytorch" />
</div>

<br>

> Bu proje, bir dijital görüntünün manipüle edilip edilmediğini hem **Geleneksel Bilgisayarlı Görü (Computer Vision)** hem de **Derin Öğrenme (AI)** tekniklerini kullanarak analiz eden bir web uygulamasıdır.

> Sistem özellikle **Copy-Move (Kopyala-Yapıştır) sahteciliğine aday bölgeleri tespit etmeye odaklanır.**

---

# 🎯 Projenin Amacı

Bu sistem:

- Görüntü içerisindeki **tekrar eden yapısal bölgeleri**
- Olası **kopyalanmış (copy-move) alanları**
- Ve piksel seviyesinde **anormallikleri**

tespit ederek kullanıcıya **şüpheli bölgeleri görsel olarak sunar**.

⚠️ Not: Sistem **kesin sahtecilik kararı vermez**, bunun yerine **şüpheli bölgeleri (candidate regions)** işaretler.

---

# 🧠 Teknik Çalışma Prensibi

Sistem iki farklı analiz katmanından oluşur:

---

## 🔹 1. Geleneksel Bilgisayarlı Görü Analizi

Kullanılan algoritmalar:

- SIFT
- SURF
- AKAZE
- ORB

Bu algoritmalar Computer Vision alanında kullanılan **feature-based yöntemlerdir**.

---

### 📌 Nasıl çalışır?

### 1. Anahtar Nokta Tespiti (Keypoints)
Görüntüdeki karakteristik bölgeler bulunur:
- köşeler
- kenarlar
- texture değişimleri

---

### 2. Descriptor (Betimleyici) Oluşturma
Her nokta için matematiksel bir "kimlik" çıkarılır.

---

### 3. Eşleşme (Matching)

Aynı görüntü içindeki noktalar karşılaştırılır:

- KNN (k=2) kullanılır
- Lowe's Ratio Test ile zayıf eşleşmeler elenir

---

### 4. Mekansal Filtreleme

- Noktalar birbirine çok yakınsa elenir
- Minimum mesafe: **50 piksel**

---

### 5. Geometrik Tutarlılık Analizi

En kritik adım:

- Eşleşmelerin vektörleri hesaplanır
- Median vektör bulunur
- Sadece bu vektöre yakın olanlar seçilir

👉 Bu, copy-move sahteciliğinin temel özelliğidir:

> “Tüm kopyalanan bölge aynı yönde ve aynı miktarda taşınır”

---

### 📊 Sonuç Mantığı

- Eğer **10'dan fazla tutarlı eşleşme** varsa:
  
👉 Görüntü **"Şüpheli"** olarak işaretlenir

---

### 🎨 Görselleştirme

Sistem:

- eşleşen noktaları
- bağlantı çizgilerini
- kaynak ve hedef bölgeleri

görsel olarak işaretler:

- 🔴 kaynak noktalar
- 🔵 hedef noktalar
- 🟢 eşleşme çizgileri
- 🟣 / 🟡 bounding box alanları

---

## 🤖 2. Yapay Zeka Analizi (CNN + LSTM)

Bu katman:

- insan gözünün fark edemediği
- piksel seviyesindeki anomalileri

tespit etmeye çalışır.

---

### Kullanılan Model:

- ResNet-18 (CNN)
- LSTM (sequence modeling)

---

### 📌 Dataset Bilgisi



**AI modeli**:

- Deepfake (yüz manipülasyonu) dataseti kullanılarak eğitilmiştir.
- Dataset **Kaggle** platformundan alınmıştır
- Dataset Link: https://www.kaggle.com/datasets/undersc0re/fake-vs-real-face-classification

<br>

**⚠️ Önemli Not**:

Bu dataset:

- Yüz sahteciliği (deepfake) odaklıdır.
- Copy-move sahteciliği için özel olarak üretilmemiştir.

<br>

👉 Bu nedenle AI modeli:

- Copy-move tespitinde destekleyici sinyal üretir.
- Ancak tek başına kesin karar vermez.


---

### 📌 Çalışma Adımları

1. Görüntü normalize edilir (224x224)
2. CNN ile feature çıkarılır
3. Feature vektörü LSTM’e verilir
4. CNN + LSTM skorları birleştirilir

---

### 📊 AI Skoru

- 0 → Temiz
- 1 → Sahte

> %50 üzeri → Şüpheli kabul edilir

---

# 🧩 Hibrit Karar Mekanizması

Final karar:

```text
Şüpheli = (Traditional) OR (AI)
```

---

## 🛠️ Proje Mimarisi
Proje, yazılım mühendisliği prensiplerine uygun olarak modüler bir yapıda geliştirilmiştir:  

<br>

```text
📦 Image Forgery Detection
│
├── 📂 main.py
├── 📂 traditional_analysis.py
├── 📂 ai_analysis.py
├── 📂 train_model.py
└── 📂 forgery-ui/
```
<br>

**main.py**: FastAPI tabanlı backend sunucusudur. İstekleri karşılar ve analiz sürecini yönetir.  

**traditional_analysis.py**: SIFT, SURF, AKAZE ve ORB gibi OpenCV tabanlı algoritmaların motorudur.  

**ai_analysis.py**: PyTorch kullanarak ResNet18 (CNN) ve LSTM modelleri ile derin analiz yapar.  

**train_model.py**: ResNet model eğitimini yapar.

**forgery-ui/**: Kullanıcı dostu React tabanlı arayüzdür.

---

## 🚀 Kurulum ve Çalıştırma

> Gereksinimler:
> - Python 3.9+ 
> - Node.js.

<br>

🐍 Backend:
```bash
pip install -r requirements.txt
python main.py
```

<br>

⚛️ Frontend:
```bash
 cd forgery-ui
 npm install
 npm start
```
---

## 👨‍💻 Geliştirici Notu
```text
Bu proje:

- eğitim amaçlıdır
- explainable AI yaklaşımı içerir
- görsel analiz sistemlerinin temelini öğretir
```