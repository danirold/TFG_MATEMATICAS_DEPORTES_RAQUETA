import cv2
import numpy as np
import os

DIR_IMAGES = "dataset/images"
DIR_MASKS_MANUAL = "dataset/train/masks_gruesas"
DIR_MASKS_AUTO = "dataset/output_lineas_auto_gruesas"

DIR_OUT_MANUAL = "dataset/overlays_manuales"
DIR_OUT_AUTO = "dataset/overlays_automaticas"

os.makedirs(DIR_OUT_MANUAL, exist_ok=True)
os.makedirs(DIR_OUT_AUTO, exist_ok=True)

def calcular_metricas(mask_true, mask_pred):
    y_true = (mask_true > 127).astype(np.uint8)
    y_pred = (mask_pred > 127).astype(np.uint8)
    
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2.0 * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) > 0 else 0.0
    
    return iou, dice

def main():
    archivos = [f for f in os.listdir(DIR_IMAGES) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Procesando {len(archivos)} imágenes para cálculo de métricas...\n")

    for nombre_img in archivos:
        base_name = os.path.splitext(nombre_img)[0]
        ruta_img = os.path.join(DIR_IMAGES, nombre_img)
        
        ruta_mask_man_png = os.path.join(DIR_MASKS_MANUAL, f"{base_name}.png")
        ruta_mask_man_orig = os.path.join(DIR_MASKS_MANUAL, nombre_img)
        ruta_mask_man = ruta_mask_man_png if os.path.exists(ruta_mask_man_png) else ruta_mask_man_orig

        ruta_mask_auto_png = os.path.join(DIR_MASKS_AUTO, f"{base_name}.png")
        ruta_mask_auto_orig = os.path.join(DIR_MASKS_AUTO, nombre_img)
        ruta_mask_auto = ruta_mask_auto_png if os.path.exists(ruta_mask_auto_png) else ruta_mask_auto_orig

        img_original = cv2.imread(ruta_img)
        mask_manual = cv2.imread(ruta_mask_man, cv2.IMREAD_GRAYSCALE)
        mask_auto = cv2.imread(ruta_mask_auto, cv2.IMREAD_GRAYSCALE)

        if img_original is None or mask_manual is None or mask_auto is None:
            print(f"Archivos faltantes para {nombre_img}, omitiendo")
            continue

        print(f"Evaluando: {nombre_img}")

        overlay_manual = img_original.copy()
        overlay_auto = img_original.copy()

        overlay_manual[mask_manual == 255] = [0, 0, 255]
        overlay_auto[mask_auto == 255] = [0, 0, 255]

        cv2.imwrite(os.path.join(DIR_OUT_MANUAL, nombre_img), overlay_manual)
        cv2.imwrite(os.path.join(DIR_OUT_AUTO, nombre_img), overlay_auto)

        iou, dice = calcular_metricas(mask_manual, mask_auto)
        print(f"   -> IoU:  {iou:.4f}")
        print(f"   -> Dice: {dice:.4f}\n")

    print(f"Proceso completado. Overlays guardados en directorios respectivos.")

if __name__ == "__main__":
    main()