# 🐾 Animal Species Classifier (Vision Transformer)

Bu proje, görüntüdeki hayvan türünü doğru şekilde tahmin edebilen bir yapay zeka modeli geliştirmeyi amaçlamaktadır. Model, **Vision Transformer (ViT)** mimarisi kullanılarak eğitilmiştir. Eğitim sonrası model `.pth` formatında kaydedilir ve bir kullanıcı arayüzü aracılığıyla tahmin işlemleri gerçekleştirilir.

---

## 📌 Proje Amacı

Görüntü sınıflandırma görevinde ViT mimarisinin performansını test ederek bir görüntüdeki hayvan türünü tahmin edebilen bir model oluşturmak. Bu model eğitim ve test işlemleri sonucunda en az %65 doğruluk sağlamalıdır. %90 ve üzeri sonuçlarda ek değerlendirme yapılır.

---

## 🧾 İsterler

Bu projede aşağıdaki gereksinimler dikkate alınmıştır:

- ✅ Görüntülerden hayvan türünü tahmin eden bir model geliştirilmeli.
- ✅ Kullanılan model **Vision Transformer (ViT)** tabanlı olmalı.
- ✅ Eğitilen model doğrulama ve test verisi üzerinde **en az %65 doğruluk** sağlamalı.
- ✅ Eğitilen model `models/` klasörüne `classifier.pth` olarak kaydedilmeli.
- ✅ Model, GUI üzerinden kullanıcıdan alınan görseli tahmin edip sonucu göstermeli.

---

## 🗂️ Proje Yapısı

├── data/
│ ├── train
│ └── test
│
├── models/
│ └── classifier.pth 
│
├── main.py 
├── gui.py 


> Not: `models/` klasörü model eğitilmeden önce oluşturulmuş olmalıdır.

---

## 🧠 Model Detayları

- 📐 **Mimari:** Vision Transformer (ViT)
- 🧾 **Çıktı:** Görüntüdeki hayvanın sınıf etiketi (örneğin: `cat`, `dog`, `bird`)
- ✅ **Başarı Kriteri:** En az %65 test doğruluğu
- 📷 **Giriş:** 224x224 boyutlu normalize RGB görüntüler

---

## 📦 Gereksinimler

Projenin çalışması için aşağıdaki kütüphaneler gereklidir:

```bash
pip install torch torchvision timm scikit-learn matplotlib pillow
