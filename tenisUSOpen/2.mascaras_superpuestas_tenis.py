import cv2
import numpy as np
import os

# Rutas
MARKED_DIR = "dataset/train/images"
MASKS_DIR = "dataset/train/masks"
OVERLAY_DIR = "dataset/train/overlays" 

os.makedirs(OVERLAY_DIR, exist_ok=True)

# Modelo 2D de la pista
puntos_modelo = {
    'tl_doubles': (286, 562),    'tr_doubles': (1379, 562),
    'bl_doubles': (286, 2935),   'br_doubles': (1379, 2935),
    'tl_singles': (423, 561),    'tr_singles': (1242, 561),
    'bl_singles': (423, 2935),   'br_singles': (1242, 2935),
    'service_tl': (423, 1110),   'service_tr': (1242, 1110),
    'service_bl': (423, 2387),   'service_br': (1242, 2386),
    'center_top': (832, 1111),   'center_btm': (832, 2386)
}

# Puntos base para la homografía (esquinas exteriores)
src_pts = np.array([
    puntos_modelo['tl_doubles'], puntos_modelo['tr_doubles'],
    puntos_modelo['bl_doubles'], puntos_modelo['br_doubles']
], dtype=np.float32)

# Array para transformar todos los puntos a la vez
nombres_puntos = list(puntos_modelo.keys())
todos_los_puntos_src = np.array([puntos_modelo[k] for k in nombres_puntos], dtype=np.float32).reshape(-1, 1, 2)

archivos = [f for f in os.listdir(MARKED_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Generando superposiciones para {len(archivos)} imágenes...")

for nombre_img in archivos:
    ruta_marcada = os.path.join(MARKED_DIR, nombre_img)
    ruta_mascara = os.path.join(MASKS_DIR, nombre_img)
    ruta_salida = os.path.join(OVERLAY_DIR, nombre_img)

    img_marcada = cv2.imread(ruta_marcada)
    mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)

    if img_marcada is None or mascara is None:
        print(f"Error al leer {nombre_img}, saltando")
        continue
    
    img_superpuesta = img_marcada.copy()
    
    # Pintar las líneas de la máscara en rojo
    img_superpuesta[mascara == 255] = [0, 0, 255] 

    # Buscar las 4 esquinas leyendo la propia máscara
    y_coords, x_coords = np.where(mascara > 0)
    
    if len(x_coords) > 0:
        # Extraer los extremos
        suma = x_coords + y_coords
        resta = x_coords - y_coords
        
        tl = (x_coords[np.argmin(suma)], y_coords[np.argmin(suma)])
        br = (x_coords[np.argmax(suma)], y_coords[np.argmax(suma)])
        tr = (x_coords[np.argmax(resta)], y_coords[np.argmax(resta)])
        bl = (x_coords[np.argmin(resta)], y_coords[np.argmin(resta)])
        
        dst_pts = np.array([tl, tr, bl, br], dtype=np.float32)
        
        # Recalcular la homografía con las esquinas extraídas
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        if H is not None:
            # Proyectar los 14 puntos del modelo
            puntos_proyectados = cv2.perspectiveTransform(todos_los_puntos_src, H)
            
            for pt in puntos_proyectados:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(img_superpuesta, (x, y), 7, (0, 0, 0), -1)
                cv2.circle(img_superpuesta, (x, y), 7, (255, 255, 255), 1)

    cv2.imwrite(ruta_salida, img_superpuesta)

print(f"Proceso completado. Guardado en {OVERLAY_DIR}")