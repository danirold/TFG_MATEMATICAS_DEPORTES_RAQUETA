import cv2
import numpy as np
import os

CARPETA = 'dataset'
NOMBRE_IMAGEN = 'pista2D.png'  
RUTA_IMAGEN = os.path.join(CARPETA, NOMBRE_IMAGEN)

FACTOR_ZOOM = 10     
TAMANO_LUPA = 60     

NOMBRES_PUNTOS = [
    "1. Esquina Superior Izq (Fondo)",
    "2. Esquina Superior Der (Fondo)",
    "3. Línea Saque Superior Izq (Pared)",
    "4. Línea Saque Superior Der (Pared)",
    "5. Red Izq (Pared)",
    "6. Red Der (Pared)",
    "7. Línea Saque Inferior Izq (Pared)",
    "8. Línea Saque Inferior Der (Pared)",
    "9. Esquina Inferior Izq (Fondo)",
    "10. Esquina Inferior Der (Fondo)",
    "11. Intersección T Superior (Centro)", 
    "12. Intersección T Inferior (Centro)"   
]

TOTAL_PUNTOS = len(NOMBRES_PUNTOS)
puntos_guardados = []
img_original = None
img_display = None

def click_zoom(event, x, y, flags, param):
    global puntos_guardados, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x_min, y_min = param['roi_origin']
        x_real = x_min + int(x / FACTOR_ZOOM)
        y_real = y_min + int(y / FACTOR_ZOOM)
        
        puntos_guardados.append((x_real, y_real))
        
        cv2.circle(img_display, (x_real, y_real), 20, (0, 0, 255), -1) 
        
        cv2.imshow('Mapa General', img_display)
        print(f"Punto {len(puntos_guardados)} guardado: ({x_real}, {y_real})")
        cv2.destroyWindow('Zoom')
        
        if len(puntos_guardados) < TOTAL_PUNTOS:
            print(f"Siguiente: {NOMBRES_PUNTOS[len(puntos_guardados)]}")
        else:
            print("Completado. Pulsa cualquier tecla para salir.")

def click_general(event, x, y, flags, param):
    global img_original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntos_guardados) >= TOTAL_PUNTOS: return

        h, w = img_original.shape[:2]
        x_min = max(0, x - TAMANO_LUPA)
        y_min = max(0, y - TAMANO_LUPA)
        x_max = min(w, x + TAMANO_LUPA)
        y_max = min(h, y + TAMANO_LUPA)

        roi = img_original[y_min:y_max, x_min:x_max]
        roi_zoom = cv2.resize(roi, (0,0), fx=FACTOR_ZOOM, fy=FACTOR_ZOOM, interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('Zoom', roi_zoom)
        cv2.moveWindow('Zoom', 100, 100)
        cv2.setMouseCallback('Zoom', click_zoom, {'roi_origin': (x_min, y_min)})
        
        print("Afina el clic en la ventana de zoom...")

def main():
    global img_original, img_display
    
    if not os.path.exists(RUTA_IMAGEN):
        print(f"Error: No se encuentra {RUTA_IMAGEN}")
        return

    img_original = cv2.imread(RUTA_IMAGEN)
    img_display = img_original.copy()

    print("Instrucciones:")
    print("1. Clic aproximado en el mapa general.")
    print("2. Clic exacto en la ventana de zoom.")
    print("-" * 20)
    print(f"Primero: {NOMBRES_PUNTOS[0]}")
    
    cv2.namedWindow('Mapa General', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mapa General', 600, 900)
    
    cv2.imshow('Mapa General', img_display)
    cv2.setMouseCallback('Mapa General', click_general)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(puntos_guardados) == TOTAL_PUNTOS:
        ruta_salida_img = os.path.join(CARPETA, 'Padel_marcados.png')
        ruta_salida_txt = os.path.join(CARPETA, 'puntos_referencia_padel.txt')
        
        cv2.imwrite(ruta_salida_img, img_display)
        
        with open(ruta_salida_txt, 'w') as f:
            f.write(f"self.p1_top_left        = {puntos_guardados[0]}\n")
            f.write(f"self.p2_top_right       = {puntos_guardados[1]}\n")
            f.write(f"self.p3_service_top_l   = {puntos_guardados[2]}\n")
            f.write(f"self.p4_service_top_r   = {puntos_guardados[3]}\n")
            f.write(f"self.p5_net_left        = {puntos_guardados[4]}\n")
            f.write(f"self.p6_net_right       = {puntos_guardados[5]}\n")
            f.write(f"self.p7_service_btm_l   = {puntos_guardados[6]}\n")
            f.write(f"self.p8_service_btm_r   = {puntos_guardados[7]}\n")
            f.write(f"self.p9_btm_left        = {puntos_guardados[8]}\n")
            f.write(f"self.p10_btm_right      = {puntos_guardados[9]}\n")
            f.write(f"self.p11_top_t_point    = {puntos_guardados[10]}\n")
            f.write(f"self.p12_bottom_t_point = {puntos_guardados[11]}\n")
            
        print(f"Guardado en {CARPETA}")
    else:
        print("Operación cancelada. Faltan puntos por marcar.")

if __name__ == "__main__":
    main()