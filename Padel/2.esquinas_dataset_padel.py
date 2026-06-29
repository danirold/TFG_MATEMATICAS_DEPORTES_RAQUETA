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

conexiones_lineas = [
    ('tl_corner', 'tr_corner'), ('bl_corner', 'br_corner'),
    ('tl_corner', 'bl_corner'), ('tr_corner', 'br_corner'),
    ('service_tl', 'service_tr'), ('service_bl', 'service_br'),
    ('t_top', 't_bottom'),
    ('net_l', 'net_r')
]

src_pts = np.array([
    puntos_modelo['tl_corner'], puntos_modelo['tr_corner'],
    puntos_modelo['bl_corner'], puntos_modelo['br_corner']
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
        cv2.imshow("Etiquetador Padel", img_display)
        print(f"Clic {len(puntos_clic)}: {x}, {y}")

archivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Procesando {len(archivos)} imágenes de pádel...")
print("Orden: Esquina Arriba-Izq, Arriba-Der, Abajo-Izq, Abajo-Der")
print("Espacio: Saltar | ESC: Salir")

cv2.namedWindow("Etiquetador Padel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Etiquetador Padel", 1280, 720)

for nombre_img in archivos:
    ruta = os.path.join(INPUT_DIR, nombre_img)
    img_actual = cv2.imread(ruta)
    if img_actual is None: continue

    img_display = img_actual.copy()
    puntos_clic = []
    
    cv2.imshow("Etiquetador Padel", img_display)
    cv2.setMouseCallback("Etiquetador Padel", click_event)
    
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

    cv2.imshow("Etiquetador Padel", mask)
    cv2.waitKey(300)

cv2.destroyAllWindows()
print("Proceso finalizado.")