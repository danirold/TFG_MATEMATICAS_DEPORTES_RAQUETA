import cv2
import numpy as np
import os

MARKED_DIR = "dataset/train/images"
MASKS_GRUESAS_DIR = "dataset/train/masks_gruesas"
MASKS_FINAS_DIR = "dataset/train/masks"
OVERLAY_DIR = "dataset/train/overlays_gruesas"

os.makedirs(OVERLAY_DIR, exist_ok=True)

puntos_modelo = {
    'L1_izq': (235, 481), 'L1_centro': (704, 481), 'L1_der': (1172, 481),
    'L2_izq': (235, 1178), 'L2_centro': (703, 1177), 'L2_der': (1173, 1178),
    'L3_izq': (235, 1504), 'L3_der': (1172, 1503),
    'L4_izq': (235, 1843), 'L4_centro': (704, 1843), 'L4_der': (1173, 1843),
    'L5_izq': (235, 2518), 'L5_centro': (704, 2518), 'L5_der': (1172, 2518)
}

src_pts = np.array([
    puntos_modelo['L1_izq'], puntos_modelo['L1_der'],
    puntos_modelo['L5_izq'], puntos_modelo['L5_der']
], dtype=np.float32)

nombres_puntos = list(puntos_modelo.keys())
todos_los_puntos_src = np.array([puntos_modelo[k] for k in nombres_puntos], dtype=np.float32).reshape(-1, 1, 2)

archivos = [f for f in os.listdir(MARKED_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Generando {len(archivos)} superposiciones...")

for nombre_img in archivos:
    ruta_marcada = os.path.join(MARKED_DIR, nombre_img)
    ruta_mascara_gruesa = os.path.join(MASKS_GRUESAS_DIR, nombre_img)
    ruta_mascara_fina = os.path.join(MASKS_FINAS_DIR, nombre_img)
    ruta_salida = os.path.join(OVERLAY_DIR, nombre_img)

    img_marcada = cv2.imread(ruta_marcada)
    mascara_gruesa = cv2.imread(ruta_mascara_gruesa, cv2.IMREAD_GRAYSCALE)
    mascara_fina = cv2.imread(ruta_mascara_fina, cv2.IMREAD_GRAYSCALE)

    if img_marcada is None or mascara_gruesa is None or mascara_fina is None:
        print(f"Error al leer archivos de {nombre_img}, omitiendo")
        continue
    
    img_superpuesta = img_marcada.copy()
    
    img_superpuesta[mascara_gruesa > 0] = [0, 0, 255] 

    y_coords, x_coords = np.where(mascara_fina > 0) 
    
    if len(x_coords) > 0:
        suma = x_coords + y_coords
        resta = x_coords - y_coords
        
        tl = (x_coords[np.argmin(suma)], y_coords[np.argmin(suma)]) 
        br = (x_coords[np.argmax(suma)], y_coords[np.argmax(suma)]) 
        tr = (x_coords[np.argmax(resta)], y_coords[np.argmax(resta)]) 
        bl = (x_coords[np.argmin(resta)], y_coords[np.argmin(resta)]) 
        
        dst_pts = np.array([tl, tr, bl, br], dtype=np.float32)
        
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        if H is not None:
            puntos_proyectados = cv2.perspectiveTransform(todos_los_puntos_src, H)
            
            for pt in puntos_proyectados:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(img_superpuesta, (x, y), 9, (0, 0, 0), -1)
                cv2.circle(img_superpuesta, (x, y), 9, (255, 255, 255), 1)

    cv2.imwrite(ruta_salida, img_superpuesta)

print(f"Proceso completado. Guardado en {OVERLAY_DIR}")