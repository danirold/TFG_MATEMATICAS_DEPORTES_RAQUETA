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
    'L1_izq': (235, 481), 'L1_centro': (704, 481), 'L1_der': (1172, 481),
    'L2_izq': (235, 1178), 'L2_centro': (703, 1177), 'L2_der': (1173, 1178),
    'L3_izq': (235, 1504), 'L3_der': (1172, 1503),
    'L4_izq': (235, 1843), 'L4_centro': (704, 1843), 'L4_der': (1173, 1843),
    'L5_izq': (235, 2518), 'L5_centro': (704, 2518), 'L5_der': (1172, 2518)
}

conexiones_lineas = [
    ('L1_izq', 'L1_der'),
    ('L5_izq', 'L5_der'),
    ('L1_izq', 'L5_izq'),
    ('L1_der', 'L5_der'),
    ('L2_izq', 'L2_der'),
    ('L4_izq', 'L4_der'),
    ('L3_izq', 'L3_der'),  
    ('L1_centro', 'L2_centro'),
    ('L4_centro', 'L5_centro')
]

src_pts = np.array([
    puntos_modelo['L1_izq'], puntos_modelo['L1_der'],
    puntos_modelo['L5_izq'], puntos_modelo['L5_der']
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
        cv2.imshow("Etiquetador Pickleball", img_display)
        print(f"Clic {len(puntos_clic)}: {x}, {y}")

archivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Procesando {len(archivos)} imágenes de pickleball...")
print("Orden: Esquina Arriba-Izq, Arriba-Der, Abajo-Izq, Abajo-Der")
print("Espacio: Saltar | ESC: Salir")

cv2.namedWindow("Etiquetador Pickleball", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Etiquetador Pickleball", 1280, 720)

for nombre_img in archivos:
    ruta = os.path.join(INPUT_DIR, nombre_img)
    img_actual = cv2.imread(ruta)
    if img_actual is None: continue

    img_display = img_actual.copy()
    puntos_clic = []
    
    cv2.imshow("Etiquetador Pickleball", img_display)
    cv2.setMouseCallback("Etiquetador Pickleball", click_event)
    
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

    cv2.imshow("Etiquetador Pickleball", mask)
    cv2.waitKey(300)

cv2.destroyAllWindows()
print("Proceso finalizado.")