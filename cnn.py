import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Periksa apakah model ada
if not os.path.exists("cnn_model.h5"):
    print("Error: Model cnn_model.h5 tidak ditemukan!")
    exit()

print("Loading model...")
# Load model
model = load_model('cnn_model.h5')
print("Model berhasil dimuat. Memulai kamera...")

# Ganti class_labels dengan nama kelas yang benar
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Buka kamera
print("Membuka kamera...")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Kamera tidak terdeteksi!")
    exit()

print("Kamera berhasil dibuka. Memulai loop utama...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera!")
        break

    print("Frame berhasil dibaca, menjalankan Night Vision...")

    # Mode Night Vision
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    print("Melakukan preprocessing gambar...")
    # Preprocessing
    img = cv2.resize(frame, (150, 150))  # Disesuaikan dengan model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Sesuaikan warna dengan dataset pelatihan
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Tambahkan pengurangan noise
    img = img.astype("float32") / 255.0
    img = (img - np.mean(img)) / (np.std(img) + 1e-7)  # Normalisasi tambahan
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    print("Melakukan prediksi...")
    # Prediksi
    pred = model.predict(img)[0]  # Ambil array prediksi pertama
    confidence_scores = {class_labels[i]: pred[i] * 100 for i in range(len(pred))}
    
    for label, score in confidence_scores.items():
        print(f"{label}: {score:.2f}%")
    
    label_index = np.argmax(pred)
    confidence = confidence_scores[class_labels[label_index]]
    
    if label_index >= len(class_labels):
        print(f"Error: Index {label_index} di luar batas class_labels {len(class_labels)}")
        label = "Unknown"
    else:
        label = class_labels[label_index]
    
    print(f"Prediksi: {label} (Index: {label_index}, Confidence: {confidence:.2f}%)")

    # Tampilkan hasil dengan confidence score
    cv2.putText(frame, f'Class: {label} ({confidence:.2f}%)', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Menutup program...")
        break

print("Melepaskan kamera dan menutup semua jendela...")
cap.release()
cv2.destroyAllWindows()
print("Program selesai.")
