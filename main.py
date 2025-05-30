import os
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

#cihazı kontrol ettik
computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA kullanımı aktif mi:", torch.cuda.is_available())

#dosya uzantılarını burada aldık ve models göndermesini ekledik
training_data_directory_path = os.path.join("data", "train")
testing_data_directory_path = os.path.join("data", "test")
model_output_save_path = os.path.join("models", "vit_base_patch16_classifier.pth")
os.makedirs("models", exist_ok=True)

#resimleri oranlama kısmı 
image_preprocessing_for_training = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image_preprocessing_for_validation_and_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#train ve dogrulama kısmı
print("Veri yükleniyor...")
complete_training_dataset = datasets.ImageFolder(training_data_directory_path, transform=image_preprocessing_for_training)

training_set_size = int(0.8 * len(complete_training_dataset))
validation_set_size = len(complete_training_dataset) - training_set_size

training_dataset, validation_dataset = random_split(complete_training_dataset, [training_set_size, validation_set_size])
validation_dataset.dataset.transform = image_preprocessing_for_validation_and_test

testing_dataset = datasets.ImageFolder(testing_data_directory_path, transform=image_preprocessing_for_validation_and_test)

list_of_class_labels = complete_training_dataset.classes

training_data_loader = DataLoader(training_dataset, batch_size=16, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
testing_data_loader = DataLoader(testing_dataset, batch_size=16, shuffle=False)

print(f"Veri yüklendi. Toplam sınıf sayısı: {len(list_of_class_labels)}")

#resmi tanımlamak için modeli tanımladık
vision_transformer_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(list_of_class_labels))
vision_transformer_model.to(computation_device)

loss_function = nn.CrossEntropyLoss()
adam_optimizer = torch.optim.Adam(vision_transformer_model.parameters(), lr=1e-4)
learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=5, gamma=0.9)

#istenilen metrik tanımlandı
def compute_classification_metrics(true_label_list, predicted_label_list):
    accuracy = accuracy_score(true_label_list, predicted_label_list)
    precision = precision_score(true_label_list, predicted_label_list, average='macro', zero_division=0)
    recall = recall_score(true_label_list, predicted_label_list, average='macro', zero_division=0)
    f1 = f1_score(true_label_list, predicted_label_list, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

#train döngüsü
def train_model_one_epoch(model_instance, data_loader, loss_function, optimizer, device_to_use):
    model_instance.train()
    total_epoch_loss = 0
    true_labels_all_batches = []
    predicted_labels_all_batches = []

    for images_batch, labels_batch in tqdm(data_loader, desc="Eğitim", leave=False):
        images_batch, labels_batch = images_batch.to(device_to_use), labels_batch.to(device_to_use)
        model_outputs = model_instance(images_batch)
        loss = loss_function(model_outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()
        predictions = torch.argmax(model_outputs, dim=1)
        predicted_labels_all_batches.extend(predictions.cpu().numpy())
        true_labels_all_batches.extend(labels_batch.cpu().numpy())

    epoch_loss_average = total_epoch_loss / len(data_loader)
    accuracy, precision, recall, f1 = compute_classification_metrics(true_labels_all_batches, predicted_labels_all_batches)
    return epoch_loss_average, accuracy, precision, recall, f1

#dogrulama
def validate_model_one_epoch(model_instance, data_loader, loss_function, device_to_use):
    model_instance.eval()
    total_epoch_loss = 0
    true_labels_all_batches = []
    predicted_labels_all_batches = []

    with torch.no_grad():
        for images_batch, labels_batch in tqdm(data_loader, desc="Doğrulama", leave=False):
            images_batch, labels_batch = images_batch.to(device_to_use), labels_batch.to(device_to_use)
            model_outputs = model_instance(images_batch)
            loss = loss_function(model_outputs, labels_batch)

            total_epoch_loss += loss.item()
            predictions = torch.argmax(model_outputs, dim=1)
            predicted_labels_all_batches.extend(predictions.cpu().numpy())
            true_labels_all_batches.extend(labels_batch.cpu().numpy())

    epoch_loss_average = total_epoch_loss / len(data_loader)
    accuracy, precision, recall, f1 = compute_classification_metrics(true_labels_all_batches, predicted_labels_all_batches)
    return epoch_loss_average, accuracy, precision, recall, f1

#burada modeli egitiyoruz artık
total_number_of_epochs = 12
early_stopping_patience_limit = 5
no_improvement_counter = 0
best_validation_f1_score_observed = 0.0

training_loss_history = []
validation_loss_history = []
training_f1_score_history = []
validation_f1_score_history = []

for current_epoch_index in range(total_number_of_epochs):
    start_time = time.time()

    train_loss, train_acc, train_prec, train_rec, train_f1 = train_model_one_epoch(
        vision_transformer_model, training_data_loader, loss_function, adam_optimizer, computation_device
    )

    val_loss, val_acc, val_prec, val_rec, val_f1 = validate_model_one_epoch(
        vision_transformer_model, validation_data_loader, loss_function, computation_device
    )

    learning_rate_scheduler.step()

    training_loss_history.append(train_loss)
    validation_loss_history.append(val_loss)
    training_f1_score_history.append(train_f1)
    validation_f1_score_history.append(val_f1)

    elapsed_time = time.time() - start_time

    print(f"Epoch {current_epoch_index + 1}/{total_number_of_epochs} - Süre: {elapsed_time:.1f}s")
    print(f"Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    if val_f1 > best_validation_f1_score_observed:
        best_validation_f1_score_observed = val_f1
        no_improvement_counter = 0
        torch.save(vision_transformer_model.state_dict(), model_output_save_path)
        print(f"Yeni en iyi model kaydedildi. (Val F1: {val_f1:.4f})")
    else:
        no_improvement_counter += 1
        print(f"Gelişme yok. Üst üste {no_improvement_counter} epoch.")

    if no_improvement_counter >= early_stopping_patience_limit:
        print(f"Erken durdurma uygulandı. {early_stopping_patience_limit} epoch boyunca gelişme gözlenmedi.")
        break

#görsel
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_loss_history, label='Eğitim Kayıp')
plt.plot(validation_loss_history, label='Doğrulama Kayıp')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_f1_score_history, label='Eğitim F1')
plt.plot(validation_f1_score_history, label='Doğrulama F1')
plt.title('F1 Skoru Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

#TEST
def evaluate_model_on_test_data(model_instance, test_data_loader, device_to_use):
    model_instance.eval()
    predicted_labels_list = []
    true_labels_list = []

    with torch.no_grad():
        for image_batch, label_batch in tqdm(test_data_loader, desc="Test Aşaması"):
            image_batch, label_batch = image_batch.to(device_to_use), label_batch.to(device_to_use)
            output_predictions = model_instance(image_batch)
            predicted_classes = torch.argmax(output_predictions, dim=1)
            predicted_labels_list.extend(predicted_classes.cpu().numpy())
            true_labels_list.extend(label_batch.cpu().numpy())

    accuracy, precision, recall, f1 = compute_classification_metrics(true_labels_list, predicted_labels_list)
    print("\n--- Test Sonuçları ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(true_labels_list, predicted_labels_list, target_names=list_of_class_labels, zero_division=0))



print("\nTest işlemi başlatılıyor...")
vision_transformer_model.load_state_dict(torch.load(model_output_save_path))
vision_transformer_model.to(computation_device)
evaluate_model_on_test_data(vision_transformer_model, testing_data_loader, computation_device)












def test_incorrect_tensor_shape():
    # Model input şekli yanlış (örneğin 3x224x224 yerine 224x224)
    dummy_image_tensor = torch.randn(224, 224).to(kullanilacak_cihaz)  
    try:
        _ = vit_modeli(dummy_image_tensor)  # Hata: Expected 4D tensor
    except Exception as e:
        print(f"test_incorrect_tensor_shape error caught: {e}")

def test_invalid_image_resize():
    # resize metoduna tuple yerine yanlış tip parametre gönderilmiş
    img = Image.new("RGB", (300, 300))
    try:
        img_resized = img.resize(224)  # Hata: tuple bekleniyor, int gönderilmiş
    except Exception as e:
        print(f"test_invalid_image_resize error caught: {e}")


def test_undefined_gui_widget_access():
    # GUI'de tanımsız değişken kullanılıyor
    try:
        undefined_label.config(text="Test")  # undefined_label tanımsız
    except Exception as e:
        print(f"test_undefined_gui_widget_access error caught: {e}")
        
        

def attempt_incorrect_tensor_prediction():
    # Model 4 boyutlu tensor beklerken 3 boyutlu tensor gönderiliyor
    dummy_image = torch.randn(3, 224, 224).to(kullanilacak_cihaz)  # batch size eksik
    try:
        output = vit_modeli(dummy_image)  # Hata: Expected 4D tensor but got 3D tensor
    except Exception as error:
        print(f"attempt_incorrect_tensor_prediction failed: {error}")



def attempt_invalid_image_resize():
    sample_image = Image.new("RGB", (256, 256))
    try:
        # resize metodu tuple (width, height) bekler, int verilmiş
        resized_image = sample_image.resize(224)
    except Exception as error:
        print(f"attempt_invalid_image_resize failed: {error}")

    try:
   
        resized_image = sample_image.resize((-100, 100))
    except Exception as error:
        print(f"attempt_invalid_image_resize failed with negative values: {error}")



def update_undefined_gui_label():
    try:
        # gorsel_etiketi tanımlanmadan önce erişilmeye çalışılıyor
        undefined_label.config(text="This will fail")
    except Exception as error:
        print(f"update_undefined_gui_label failed: {error}")


