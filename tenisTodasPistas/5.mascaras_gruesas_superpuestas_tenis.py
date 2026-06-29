import cv2
import numpy as np
import os

MARKED_DIR = "dataset/train_engrosar2/images"
MASKS_GRUESAS_DIR = "dataset/train_engrosar2/masks_gruesas"
MASKS_FINAS_DIR = "dataset/train_engrosar2/masks"
OVERLAY_DIR = "dataset/train_engrosar2/overlays" 

os.makedirs(OVERLAY_DIR, exist_ok=True)

puntos_modelo = {
    'tl_doubles': (286, 562),    'tr_doubles': (1379, 562),
    'bl_doubles': (286, 2935),   'br_doubles': (1379, 2935),
    'tl_singles': (423, 561),    'tr_singles': (1242, 561),
    'bl_singles': (423, 2935),   'br_singles': (1242, 2935),
    'service_tl': (423, 1110),   'service_tr': (1242, 1110),
    'service_bl': (423, 2387),   'service_br': (1242, 2386),
    'center_top': (832, 1111),   'center_btm': (832, 2386)
}

src_pts = np.array([
    puntos_modelo['tl_doubles'], puntos_modelo['tr_doubles'],
    puntos_modelo['bl_doubles'], puntos_modelo['br_doubles']
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
    
    # Pintar máscara gruesa en rojo
    img_superpuesta[mascara_gruesa == 255] = [0, 0, 255] 

    # Extraer esquinas usando la máscara fina
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