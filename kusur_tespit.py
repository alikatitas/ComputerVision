import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- AYARLAR ---
ANA_PATH = "fruit-360-dataset" 

# Kusur tespiti senaryosu için birbirine benzeyen iki sınıf seçiyoruz.
# Sağlam Meyve ile Çürük Meyve ayırımı yapacağız

SAGLAM_SINIF = "Tomato Cherry Orange 1"
KUSURLU_SINIF = "Tomato Cherry Maroon 1" 

def ozellik_cikar(goruntu_yolu):
    img = cv2.imread(goruntu_yolu)
    if img is None: return None
    
    # 1. LAB Renk Analizi (Renk bozulmalarını yakalar - Leke, Çürük)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    
    # Renk istatistikleri (Ortalama, Standart Sapma, Çarpıklık)
    # Çürük bölgeler genellikle renk homojenliğini bozar, bu yüzden std sapma önemlidir.
    l_mean, l_std = np.mean(l), np.std(l)
    a_mean, a_std = np.mean(a), np.std(a)
    b_mean, b_std = np.mean(b), np.std(b)

    # 2. GLCM Doku Analizi (Yüzey bozulmalarını yakalar - Buruşma, Yara)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # distance=[11] piksel komşuluğu, yüzeydeki ince detayları yakalar.
    glcm = graycomatrix(gray_img, distances=[11], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    
    # Tüm özellikleri birleştir
    return [l_mean, l_std, a_mean, a_std, b_mean, b_std, contrast, energy, homogeneity, correlation]

def veri_yukle_ikili(ana_yol, tip, sinif1, sinif2):
    path = os.path.join(ana_yol, tip)
    data = []
    labels = []
    
    # Sınıf 1 Yükleme (Sağlam)
    path1 = os.path.join(path, sinif1)
    if os.path.isdir(path1):
        print(f"{tip} verisi yükleniyor: {sinif1} (Sağlam olarak etiketlendi)...")
        for f in os.listdir(path1):
            feat = ozellik_cikar(os.path.join(path1, f))
            if feat:
                data.append(feat)
                labels.append(0) # 0: Sağlam

    # Sınıf 2 Yükleme (Kusurlu)
    path2 = os.path.join(path, sinif2)
    if os.path.isdir(path2):
        print(f"{tip} verisi yükleniyor: {sinif2} (Kusurlu olarak etiketlendi)...")
        for f in os.listdir(path2):
            feat = ozellik_cikar(os.path.join(path2, f))
            if feat:
                data.append(feat)
                labels.append(1) # 1: Kusurlu/Farklı
                
    return np.array(data), np.array(labels)

# --- ANA İŞLEM ---
print("--- KUSUR TESPİTİ / KALİTE SINIFLANDIRMA SİMÜLASYONU ---")

# Eğitim Verisi
X_train, y_train = veri_yukle_ikili(ANA_PATH, "Training", SAGLAM_SINIF, KUSURLU_SINIF)

# Test Verisi
X_test, y_test = veri_yukle_ikili(ANA_PATH, "Test", SAGLAM_SINIF, KUSURLU_SINIF)

print(f"\nEğitim Seti: {len(X_train)} görüntü")
print(f"Test Seti: {len(X_test)} görüntü")

# Model Eğitimi (SVM)
# Kusur tespiti için RBF kernel bazen Lineer'den daha iyi sonuç verebilir çünkü kusurlar lineer olmayabilir.
print("\nSVM Modeli (Kernel='rbf') Eğitiliyor...")
clf = svm.SVC(kernel='rbf', C=10, gamma='scale') 
clf.fit(X_train, y_train)

# Tahmin
y_pred = clf.predict(X_test)

# Sonuçlar
acc = accuracy_score(y_test, y_pred)
print(f"\nAYIRT ETME BAŞARISI: %{acc * 100:.2f}")

print("\nDetaylı Rapor:")
target_names = ['Sağlam (Sınıf 1)', 'Kusurlu/Farklı (Sınıf 2)']
print(classification_report(y_test, y_pred, target_names=target_names))

# Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
plt.title(f'{SAGLAM_SINIF} vs {KUSURLU_SINIF} Ayrımı')
plt.ylabel('Gerçek Durum')
plt.xlabel('Tahmin Edilen')
plt.show()