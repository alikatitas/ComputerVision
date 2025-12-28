import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- 1. DOSYA YOLU AYARLARI ---
# Zip'ten çıkardığınız ana klasörün yolu.
# Klasör yapısının şuna benzediğini varsayıyoruz: 
#   .../fruit-360/Training
#   .../fruit-360/Test
ANA_PATH = "fruit-360-dataset" 

# --- 2. SINIFLARI OTOMATİK BELİRLEME ---
train_dir = os.path.join(ANA_PATH, "Training")
if not os.path.exists(train_dir):
    print(f"HATA: '{train_dir}' klasörü bulunamadı. Lütfen ANA_PATH yolunu kontrol edin.")
    exit()

# Training klasöründeki tüm alt klasör isimlerini (meyve adlarını) alıyoruz
# Eğer sadece belirli meyveleri istiyorsanız buraya liste olarak yazabilirsiniz.
TUM_SINIFLAR = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print(f"Tespit Edilen Sınıf Sayısı: {len(TUM_SINIFLAR)}")
# Örnek olarak ilk 5 sınıfı yazdıralım
print(f"Örnek Sınıflar: {TUM_SINIFLAR[:5]} ...")

# --- 3. ÖZNİTELİK ÇIKARMA (LAB + GLCM) ---
def ozellik_cikar(goruntu_yolu):
    img = cv2.imread(goruntu_yolu)
    if img is None: return None
    
    # LAB Renk Uzayı Analizi
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l_mean, l_std = np.mean(l), np.std(l)
    a_mean, a_std = np.mean(a), np.std(a)
    b_mean, b_std = np.mean(b), np.std(b)

    # GLCM Doku Analizi
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))

    return [l_mean, l_std, a_mean, a_std, b_mean, b_std, contrast, dissimilarity, homogeneity, energy, correlation]

# --- 4. VERİ YÜKLEME FONKSİYONU ---
def klasor_yukle(ana_yol, veri_tipi, sinif_listesi):
    """
    veri_tipi: 'Training' veya 'Test'
    Bu fonksiyon belirtilen klasördeki resimleri okur ve özelliklerini çıkarır.
    """
    hedef_klasor = os.path.join(ana_yol, veri_tipi)
    data = []
    labels = []
    
    print(f"\n--- {veri_tipi} Verileri Yükleniyor ---")
    
    # İlerleme durumunu göstermek için sayaç
    start = time.time()
    for i, sinif in enumerate(sinif_listesi):
        sinif_yolu = os.path.join(hedef_klasor, sinif)
        
        if not os.path.isdir(sinif_yolu):
            continue
            
        # Sadece resim dosyalarını al
        resimler = [f for f in os.listdir(sinif_yolu) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Her sınıftaki tüm resimleri işle
        for dosya in resimler:
            ozellikler = ozellik_cikar(os.path.join(sinif_yolu, dosya))
            if ozellikler is not None:
                data.append(ozellikler)
                labels.append(sinif)
        
        # Kullanıcıyı bilgilendir (Her 10 sınıfta bir veya toplam sayı azsa her adımda)
        if (i+1) % 10 == 0 or (i+1) == len(sinif_listesi):
            print(f"[{i+1}/{len(sinif_listesi)}] '{sinif}' sınıfı işlendi. (Geçen süre: {time.time()-start:.1f} sn)")
            
    return data, labels

# --- 5. ANA İŞLEM AKIŞI ---

# A) EĞİTİM (TRAINING) VERİSİNİ YÜKLE
# Model bu verileri kullanarak "öğrenecek".
X_train, y_train = klasor_yukle(ANA_PATH, "Training", TUM_SINIFLAR)
print(f"-> Toplam Eğitim Verisi: {len(X_train)} adet.")

# B) TEST VERİSİNİ YÜKLE
# Modelin başarısını ölçmek için bu verileri "karşılaştırma" amaçlı kullanacağız.
X_test, y_test = klasor_yukle(ANA_PATH, "Test", TUM_SINIFLAR)
print(f"-> Toplam Test Verisi: {len(X_test)} adet.")

# C) MODEL EĞİTİMİ
print("\nSVM Modeli 'Training' verisi ile eğitiliyor...")
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# D) KARŞILAŞTIRMA VE TEST
print("\n'Test' verisi üzerinde tahmin yapılıyor...")
y_pred = clf.predict(X_test)

# E) SONUÇLARIN RAPORLANMASI
dogruluk = accuracy_score(y_test, y_pred)
print(f"\n==========================================")
print(f"GENEL BAŞARI (Accuracy): %{dogruluk * 100:.2f}")
print(f"==========================================")

# Sınıf sayısı 20'den azsa detaylı rapor ve matris çizdir, çoksa sadece özeti göster.
if len(TUM_SINIFLAR) <= 20:
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    # Karmaşıklık Matrisi (Önceki görseldeki gibi [1])
    cm = confusion_matrix(y_test, y_pred, labels=TUM_SINIFLAR)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=TUM_SINIFLAR, yticklabels=TUM_SINIFLAR, cmap='Blues')
    plt.title('Eğitim vs Test Verisi Karşılaştırması (Confusion Matrix)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Sınıf (Test Verisi)')
    plt.show()
else:
    print("\nÇok fazla sınıf olduğu için grafik çizdirilmedi.")
    print("Ancak genel doğruluk oranı, Test klasöründeki verilerle Training verilerinin başarısını yansıtır.")