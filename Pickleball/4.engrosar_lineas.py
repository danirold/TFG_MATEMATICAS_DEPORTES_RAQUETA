import cv2
import os
import numpy as np

carpeta_entrada = "dataset/train/masks"
carpeta_salida = "dataset/train/masks_gruesas" 

TAMANO_PINCEL = (3, 3) 
ITERACIONES = 1

if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

kernel = np.ones(TAMANO_PINCEL, np.uint8)

archivos = [f for f in os.listdir(carpeta_entrada) if f.endswith(('.png', '.jpg', '.jpeg'))]
total_archivos = len(archivos)

if total_archivos == 0:
    print(f"No se encontraron imágenes en {carpeta_entrada}")
    exit()

print(f"Iniciando engrosamiento de {total_archivos} máscaras...")

procesadas = 0

for nombre_archivo in archivos:
    ruta_input = os.path.join(carpeta_entrada, nombre_archivo)
    ruta_output = os.path.join(carpeta_salida, nombre_archivo)
    
    mascara = cv2.imread(ruta_input, cv2.IMREAD_GRAYSCALE)
    _, mascara_binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
    
    mascara_engrosada = cv2.dilate(mascara_binaria, kernel, iterations=ITERACIONES)
    
    cv2.imwrite(ruta_output, mascara_engrosada)
    procesadas += 1

print(f"Completado: {procesadas} máscaras engrosadas en {carpeta_salida}")