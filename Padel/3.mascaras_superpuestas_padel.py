import cv2
import numpy as np
import os

MARKED_DIR = "dataset/train/images"
MASKS_DIR = "dataset/train/masks"
OVERLAY_DIR = "dataset/train/overlays"

os.makedirs(OVERLAY_DIR, exist_ok=True)

puntos_modelo = {
    'tl_corner': (229, 486),   
    'tr_corner': (1178, 487),  
    'bl_corner': (229, 2545),  
    'br_corner': (1178, 2545), 
    'service_tl': (230, 783),
    'service_tr': (1177, 783),
    'net_l': (229, 1519),
    'net_r': (1178, 1519),
    'service_bl': (230, 2254),
    'service_br': (1178, 2254),
    't_top': (703, 783),
    't_bottom': (704, 2253)
}

src_pts = np.array([
    puntos_modelo['tl_corner'], puntos_modelo['tr_corner'],
    puntos_modelo['bl_corner'], puntos_modelo['br_corner']
], dtype=np.float32)

nombres_puntos = list(puntos_modelo.keys())
todos_los_puntos_src = np.array([puntos_modelo[k] for k in nombres_puntos], dtype=np.float32).reshape(-1, 1, 2)

archivos = [f for f in os.listdir(MARKED_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Generando {len(archivos)} superposiciones...")

for nombre_img in archivos:
    ruta_marcada = os.path.join(MARKED_DIR, nombre_img)
    ruta_mascara = os.path.join(MASKS_DIR, nombre_img)
    ruta_salida = os.path.join(OVERLAY_DIR, nombre_img)

    img_marcada = cv2.imread(ruta_marcada)
    mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)

    if img_marcada is None or mascara is None:
        print(f"Error al leer {nombre_img}, omitiendo")
        continue
    
    img_superpuesta = img_marcada.copy()
    
    img_superpuesta[mascara == 255] = [0, 0, 255] 

    y_coords, x_coords = np.where(mascara > 0)
    
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
                cv2.circle(img_superpuesta, (x, y), 7, (0, 0, 0), -1)
                cv2.circle(img_superpuesta, (x, y), 7, (255, 255, 255), 1)

    cv2.imwrite(ruta_salida, img_superpuesta)

print(f"Proceso completado. Guardado en {OVERLAY_DIR}")