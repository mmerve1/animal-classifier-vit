import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tkinter import messagebox
from torchvision import datasets
from torch.utils.data import DataLoader
import os

# modeli aldık
model_dosya_yolu = os.path.join("models", "classifier.pth")
sinif_isimleri = sorted(os.listdir("data/train"))  # Klasör adları sınıf isimleri olarak kullanılır

goruntu_donusumu = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# model yükleme kısmı
kullanilacak_cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sinif_sayisi = len(sinif_isimleri)
vit_modeli = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=sinif_sayisi)
vit_modeli.load_state_dict(torch.load(model_dosya_yolu, map_location=kullanilacak_cihaz))
vit_modeli.eval().to(kullanilacak_cihaz)

# tek resim tahmini
def resim_tahmini_yap(resim_yolu):
    resim = Image.open(resim_yolu).convert("RGB")
    resim_tensor = goruntu_donusumu(resim).unsqueeze(0).to(kullanilacak_cihaz)

    with torch.no_grad():
        model_cikti = vit_modeli(resim_tensor)
        olasiliklar = torch.nn.functional.softmax(model_cikti, dim=1)
        guven, tahmin_indeksi = torch.max(olasiliklar, 1)

    tahmin_edilen_sinif = sinif_isimleri[tahmin_indeksi.item()]
    return tahmin_edilen_sinif, guven.item(), resim

def resim_sec_ve_tahmin_et():
    secilen_dosya = filedialog.askopenfilename()
    if secilen_dosya:
        sinif_adi, guven_degeri, orijinal_resim = resim_tahmini_yap(secilen_dosya)

        yeniden_boyutlanmis_resim = orijinal_resim.resize((200, 200))
        gorsel_tk_formatinda = ImageTk.PhotoImage(yeniden_boyutlanmis_resim)

        gorsel_etiketi.config(image=gorsel_tk_formatinda)
        gorsel_etiketi.image = gorsel_tk_formatinda

        tahmin_sonucu_metin.set(f"Tahmin Edilen Sınıf: {sinif_adi}\nGüven Oranı: {guven_degeri*100:.2f}%")

# test klasörü tarama ve başarı oranlarını hesaplama
def tum_test_klasorunu_degerlendir():
    test_klasoru = filedialog.askdirectory(title="Test klasörünü seçin")
    if not test_klasoru:
        return

    test_dataset = datasets.ImageFolder(test_klasoru, transform=goruntu_donusumu)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(kullanilacak_cihaz), labels.to(kullanilacak_cihaz)
            outputs = vit_modeli(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    total = len(y_true)
    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    messagebox.showinfo(
        "Test Sonuçları",
        f"Toplam Görsel: {total}\n"
        f"Doğru Tahmin: {correct} / {total}\n"
        f"Doğruluk (Accuracy): {acc:.4f}\n"
        f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}"
    )

# GUI arayüz
ana_pencere = tk.Tk()
ana_pencere.title("Görüntü Sınıflandırma Arayüzü")
ana_pencere.geometry("420x500")
ana_pencere.resizable(False, False)

resim_sec_butonu = tk.Button(
    ana_pencere,
    text="Resim Seç ve Tahmin Et",
    font=("Arial", 12, "bold"),
    command=resim_sec_ve_tahmin_et,
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5
)
resim_sec_butonu.pack(pady=10)

tum_test_butonu = tk.Button(
    ana_pencere,
    text="Test Klasörü Değerlendir",
    font=("Arial", 12, "bold"),
    command=tum_test_klasorunu_degerlendir,
    bg="#2196F3",
    fg="white",
    padx=10,
    pady=5
)
tum_test_butonu.pack(pady=5)

gorsel_etiketi = Label(ana_pencere)
gorsel_etiketi.pack()

tahmin_sonucu_metin = tk.StringVar()
tahmin_etiketi = Label(ana_pencere, textvariable=tahmin_sonucu_metin, font=("Arial", 12), wraplength=380, justify="center")
tahmin_etiketi.pack(pady=10)

ana_pencere.mainloop()
