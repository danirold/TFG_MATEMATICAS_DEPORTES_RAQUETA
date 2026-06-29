import cv2
import numpy as np
import os

INPUT_DIR = "dataset/images"
OUTPUT_IMG_DIR = "dataset/train/images"
OUTPUT_MASK_DIR = "dataset/train/masks"
OUTPUT_MARKED_DIR = "dataset/train/marked"

GROSOR_LINEA = 4

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_MARKED_DIR, exist_ok=True)

# Coordenadas base del modelo 2D
puntos_modelo = {
    'tl_doubles': (286, 562),    'tr_doubles': (1379, 562),
    'bl_doubles': (286, 2935),   'br_doubles': (1379, 2935),
    'tl_singles': (423, 561),    'tr_singles': (1242, 561),
    'bl_singles': (423, 2935),   'br_singles': (1242, 2935),
    'service_tl': (423, 1110),   'service_tr': (1242, 1110),
    'service_bl': (423, 2387),   'service_br': (1242, 2386),
    'center_top': (832, 1111),   'center_btm': (832, 2386)
}

conexiones_lineas = [
    ('tl_doubles', 'tr_doubles'), ('bl_doubles', 'br_doubles'),
    ('tl_doubles', 'bl_doubles'), ('tr_doubles', 'br_doubles'),
    ('tl_singles', 'bl_singles'), ('tr_singles', 'br_singles'),
    ('service_tl', 'service_tr'), ('service_bl', 'service_br'),
    ('center_top', 'center_btm')
]

# Puntos para la homografía (esquinas exteriores)
src_pts = np.array([
    puntos_modelo['tl_doubles'], puntos_modelo['tr_doubles'],
    puntos_modelo['bl_doubles'], puntos_modelo['br_doubles']
], dtype=np.float32)

puntos_clic = []
img_actual = None
img_display = None

def click_event(event, x, y, flags, param):
    global puntos_clic, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntos_clic) >= 4: return
        
        puntos_clic.append((x, y))
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow("Etiquetador", img_display)
        print(f"Clic {len(puntos_clic)}: {x}, {y}")

archivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Procesando {len(archivos)} imágenes...")
print("Orden: Arriba-Izq, Arriba-Der, Abajo-Izq, Abajo-Der")
print("Espacio: Saltar | ESC: Salir")

cv2.namedWindow("Etiquetador", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Etiquetador", 1280, 720)

for nombre_img in archivos:
    ruta = os.path.join(INPUT_DIR, nombre_img)
    img_actual = cv2.imread(ruta)
    if img_actual is None: continue

    img_display = img_actual.copy()
    puntos_clic = []
    
    cv2.imshow("Etiquetador", img_display)
    cv2.setMouseCallback("Etiquetador", click_event)
    
    saltar = False
    while len(puntos_clic) < 4:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            exit()
        elif key == 32:
            print(f"Saltando {nombre_img}")
            saltar = True
            break
    
    if saltar: continue

    print(f"Guardando {nombre_img}...")
    
    # Cálculo de homografía y transformación de puntos
    dst_pts = np.array(puntos_clic, dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    nombres = list(puntos_modelo.keys())
    coords = [puntos_modelo[k] for k in nombres]
    pts_src_todos = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
    pts_trans = cv2.perspectiveTransform(pts_src_todos, H)
    
    puntos_foto = {n: (int(p[0][0]), int(p[0][1])) for n, p in zip(nombres, pts_trans)}

    h, w = img_actual.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for (inicio, fin) in conexiones_lineas:
        pt1 = puntos_foto[inicio]
        pt2 = puntos_foto[fin]
        cv2.line(mask, pt1, pt2, 255, GROSOR_LINEA, cv2.LINE_AA)

    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, nombre_img), img_actual)
    cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, nombre_img), mask)
    cv2.imwrite(os.path.join(OUTPUT_MARKED_DIR, nombre_img), img_display)

    cv2.imshow("Etiquetador", mask)
    cv2.waitKey(300)

cv2.destroyAllWindows()
print("Proceso finalizado.")