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
    'L1_ext_izq': (235, 482), 'L1_int_izq': (315, 482), 'L1_centro': (703, 482), 'L1_int_der': (1093, 482), 'L1_ext_der': (1172, 482),
    'L2_ext_izq': (235, 618), 'L2_int_izq': (315, 618), 'L2_centro': (704, 619), 'L2_int_der': (1093, 619), 'L2_ext_der': (1173, 618),
    'L3_ext_izq': (235, 1179), 'L3_int_izq': (315, 1179), 'L3_centro': (704, 1178), 'L3_int_der': (1093, 1179), 'L3_ext_der': (1172, 1179),
    'L4_ext_izq': (235, 1845), 'L4_int_izq': (315, 1845), 'L4_centro': (704, 1845), 'L4_int_der': (1093, 1845), 'L4_ext_der': (1172, 1845),
    'L5_ext_izq': (235, 2392), 'L5_int_izq': (315, 2392), 'L5_centro': (704, 2391), 'L5_int_der': (1093, 2392), 'L5_ext_der': (1172, 2391),
    'L6_ext_izq': (235, 2520), 'L6_int_izq': (315, 2521), 'L6_centro': (704, 2521), 'L6_int_der': (1093, 2521), 'L6_ext_der': (1172, 2520)
}

conexiones_lineas = [
    ('L1_ext_izq', 'L1_ext_der'),
    ('L2_ext_izq', 'L2_ext_der'),
    ('L3_ext_izq', 'L3_ext_der'),
    ('L4_ext_izq', 'L4_ext_der'),
    ('L5_ext_izq', 'L5_ext_der'),
    ('L6_ext_izq', 'L6_ext_der'),
    ('L1_ext_izq', 'L6_ext_izq'),
    ('L1_int_izq', 'L6_int_izq'),
    ('L1_int_der', 'L6_int_der'),
    ('L1_ext_der', 'L6_ext_der'),
    ('L1_centro', 'L3_centro'),
    ('L4_centro', 'L6_centro')
]

src_pts = np.array([
    puntos_modelo['L1_ext_izq'], puntos_modelo['L1_ext_der'],
    puntos_modelo['L6_ext_izq'], puntos_modelo['L6_ext_der']
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
        cv2.imshow("Etiquetador Badminton", img_display)
        print(f"Clic {len(puntos_clic)}: {x}, {y}")

archivos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"Procesando {len(archivos)} imágenes...")
print("Orden: Esquina Arriba-Izq, Arriba-Der, Abajo-Izq, Abajo-Der")
print("Espacio: Saltar | ESC: Salir")

cv2.namedWindow("Etiquetador Badminton", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Etiquetador Badminton", 1280, 720)

for nombre_img in archivos:
    ruta = os.path.join(INPUT_DIR, nombre_img)
    img_actual = cv2.imread(ruta)
    if img_actual is None: continue

    img_display = img_actual.copy()
    puntos_clic = []
    
    cv2.imshow("Etiquetador Badminton", img_display)
    cv2.setMouseCallback("Etiquetador Badminton", click_event)
    
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

    cv2.imshow("Etiquetador Badminton", mask)
    cv2.waitKey(300)

cv2.destroyAllWindows()
print("Proceso finalizado.")