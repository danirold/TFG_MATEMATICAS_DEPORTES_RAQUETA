import cv2
import numpy as np
import os

MARKED_DIR = "dataset/train/images"
MASKS_DIR = "dataset/train/masks"
OVERLAY_DIR = "dataset/train/overlays" 

os.makedirs(OVERLAY_DIR, exist_ok=True)

puntos_modelo = {
    'L1_ext_izq': (235, 482), 'L1_int_izq': (315, 482), 'L1_centro': (703, 482), 'L1_int_der': (1093, 482), 'L1_ext_der': (1172, 482),
    'L2_ext_izq': (235, 618), 'L2_int_izq': (315, 618), 'L2_centro': (704, 619), 'L2_int_der': (1093, 619), 'L2_ext_der': (1173, 618),
    'L3_ext_izq': (235, 1179), 'L3_int_izq': (315, 1179), 'L3_centro': (704, 1178), 'L3_int_der': (1093, 1179), 'L3_ext_der': (1172, 1179),
    'L4_ext_izq': (235, 1845), 'L4_int_izq': (315, 1845), 'L4_centro': (704, 1845), 'L4_int_der': (1093, 1845), 'L4_ext_der': (1172, 1845),
    'L5_ext_izq': (235, 2392), 'L5_int_izq': (315, 2392), 'L5_centro': (704, 2391), 'L5_int_der': (1093, 2392), 'L5_ext_der': (1172, 2391),
    'L6_ext_izq': (235, 2520), 'L6_int_izq': (315, 2521), 'L6_centro': (704, 2521), 'L6_int_der': (1093, 2521), 'L6_ext_der': (1172, 2520)
}

src_pts = np.array([
    puntos_modelo['L1_ext_izq'], puntos_modelo['L1_ext_der'],
    puntos_modelo['L6_ext_izq'], puntos_modelo['L6_ext_der']
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
    
    img_superpuesta[mascara > 0] = [0, 0, 255] 

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