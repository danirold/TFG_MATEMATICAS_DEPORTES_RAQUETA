import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Rutas
IMAGES_DIR = os.path.join("dataset", "images")
OUTPUT_KMEANS_DIR = os.path.join("dataset", "kMeans")
OUTPUT_MASK_PISTA_DIR = os.path.join("dataset", "masks_pista_ruido")

os.makedirs(OUTPUT_KMEANS_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_PISTA_DIR, exist_ok=True)

IMG_NAMES = ["img1.png", "img2.png", "img3.png"]

def segmentar_kmeans(img, K=3):
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    segmented = centers[labels].reshape(img.shape)
    return segmented.astype(np.uint8), labels, centers

def obtener_mascara_super_limpia(img, K=3, area_threshold_ratio=0.05):
    seg_img, labels, centers = segmentar_kmeans(img, K)

    # El cluster de la pista suele ser el más azul
    blue_values = [center[2] for center in centers]
    pista_cluster = np.argmax(blue_values)

    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == pista_cluster] = 255
    mask = mask.reshape(img.shape[:2])

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Filtrado por contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask)
    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        if cv2.contourArea(cnt) >= img_area * area_threshold_ratio:
            cv2.drawContours(mask_final, [cnt], -1, 255, thickness=cv2.FILLED)

    return seg_img, mask_final

for img_name in IMG_NAMES:
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error al cargar {img_name}, saltando")
        continue

    print(f"Procesando {img_name}...")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]

    # Escalado para acelerar KMeans
    scale = 0.4
    img_small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))

    seg_k3, mask_clean_small = obtener_mascara_super_limpia(img_small, K=3)
    mask_clean = cv2.resize(mask_clean_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    kmeans_path = os.path.join(OUTPUT_KMEANS_DIR, f"{os.path.splitext(img_name)[0]}_kmeans.png")
    mask_path = os.path.join(OUTPUT_MASK_PISTA_DIR, f"{os.path.splitext(img_name)[0]}_pista_ruido.png")

    cv2.imwrite(kmeans_path, cv2.cvtColor(seg_k3, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask_clean)

print("Procesamiento completado.")