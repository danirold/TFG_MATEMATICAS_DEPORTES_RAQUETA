import cv2
import numpy as np
import os
import glob

DIR_MASKS_PISTA = "dataset/masks_pista"
DIR_OUTPUT_AUTO = "dataset/output_lineas_auto"

os.makedirs(DIR_OUTPUT_AUTO, exist_ok=True)

GROSOR_LINEA = 4

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

src_pts = np.array([
    puntos_modelo['tl_doubles'], puntos_modelo['tr_doubles'],
    puntos_modelo['bl_doubles'], puntos_modelo['br_doubles']
], dtype=np.float32)


def ordenar_esquinas(pts):
    pts = pts.reshape((4, 2))
    pts = pts[np.argsort(pts[:, 1])]
    
    top = pts[:2]
    bottom = pts[2:]
    
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    
    return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

def obtener_esquinas_mascara(mask_img):
    _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
        
    c_max = max(contornos, key=cv2.contourArea)
    peri = cv2.arcLength(c_max, True)
    
    # Tolerancia geométrica para aproximar a un polígono
    approx = cv2.approxPolyDP(c_max, 0.02 * peri, True)
    
    if len(approx) == 4:
        return ordenar_esquinas(approx)
    return None


def main():
    archivos_pista = glob.glob(os.path.join(DIR_MASKS_PISTA, "*.*"))
    print(f"Procesando {len(archivos_pista)} máscaras...")
    
    for ruta_pista in archivos_pista:
        nombre_archivo = os.path.basename(ruta_pista)
        
        mask_pista = cv2.imread(ruta_pista, cv2.IMREAD_GRAYSCALE)
        if mask_pista is None: 
            print(f"Error al leer: {nombre_archivo}")
            continue
            
        h, w = mask_pista.shape
        
        dst_pts = obtener_esquinas_mascara(mask_pista)
        
        if dst_pts is None:
            print(f"Omitiendo {nombre_archivo}: No se extrajeron 4 esquinas exactas")
            continue
            
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        # Transformar coordenadas y mapear
        nombres = list(puntos_modelo.keys())
        coords = [puntos_modelo[k] for k in nombres]
        pts_src_todos = np.array(coords, dtype=np.float32).reshape(-1, 1, 2)
        pts_trans = cv2.perspectiveTransform(pts_src_todos, H)
        
        puntos_foto = {n: (int(p[0][0]), int(p[0][1])) for n, p in zip(nombres, pts_trans)}
        
        # Generar máscara de salida
        auto_lines_mask = np.zeros((h, w), dtype=np.uint8)
        for (inicio, fin) in conexiones_lineas:
            pt1 = puntos_foto[inicio]
            pt2 = puntos_foto[fin]
            cv2.line(auto_lines_mask, pt1, pt2, 255, GROSOR_LINEA, cv2.LINE_AA)
            
        ruta_salida = os.path.join(DIR_OUTPUT_AUTO, nombre_archivo)
        cv2.imwrite(ruta_salida, auto_lines_mask)
            
    print("Extracción de líneas finalizada.")

if __name__ == "__main__":
    main()