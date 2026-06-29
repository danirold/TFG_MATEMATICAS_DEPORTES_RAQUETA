import cv2
import numpy as np
import os

INPUT_DIR = "dataset/masks_pista_ruido"
OUTPUT_DIR = "dataset/masks_pista"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Iniciando limpieza de máscaras...")

for mask_name in os.listdir(INPUT_DIR):
    if not mask_name.lower().endswith((".png", ".jpg")):
        continue

    input_path = os.path.join(INPUT_DIR, mask_name)
    mask_ruidosa = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if mask_ruidosa is None:
        print(f"Error al leer {mask_name}, saltando")
        continue

    # Buscar el contorno externo más grande
    contours, _ = cv2.findContours(mask_ruidosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No se detectaron contornos en {mask_name}")
        continue
        
    c = max(contours, key=cv2.contourArea)

    # Aplicar convex hull para rellenar irregularidades
    hull = cv2.convexHull(c)

    h, w = mask_ruidosa.shape[:2]
    mask_limpia = np.zeros((h, w), dtype=np.uint8)

    # Dibujar el resultado
    cv2.fillPoly(mask_limpia, [hull], 255)

    base_name = mask_name.split('_')[0].split('.')[0]
    output_name = f"{base_name}_pista.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    cv2.imwrite(output_path, mask_limpia)

print("Proceso completado.")