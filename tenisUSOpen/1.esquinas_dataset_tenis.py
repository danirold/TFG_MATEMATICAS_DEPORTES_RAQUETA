import cv2
import numpy as np
import os

CARPETA = 'dataset'
NOMBRE_IMAGEN = 'pista2D.png'
RUTA_IMAGEN = os.path.join(CARPETA, NOMBRE_IMAGEN)

FACTOR_ZOOM = 10
TAMANO_LUPA = 60

# Puntos a marcar en orden
NOMBRES_PUNTOS = [
    "1. Esquina Superior Izq (DOBLES)",
    "2. Esquina Superior Der (DOBLES)",
    "3. Esquina Superior Izq (SINGLES)",
    "4. Esquina Superior Der (SINGLES)",
    "5. Saque Superior Izq (Linea T)",
    "6. Saque Superior Der (Linea T)",
    "7. Saque Inferior Izq (Linea T)",
    "8. Saque Inferior Der (Linea T)",
    "9. Esquina Inferior Izq (SINGLES)",
    "10. Esquina Inferior Der (SINGLES)",
    "11. Esquina Inferior Izq (DOBLES)",
    "12. Esquina Inferior Der (DOBLES)",
    "13. INTERSECCION T SUPERIOR (Centro)", 
    "14. INTERSECCION T INFERIOR (Centro)"   
]

puntos_guardados = []
img_original = None
img_display = None

def click_zoom(event, x, y, flags, param):
    # Registra el clic en la ventana ampliada
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
        
        if len(puntos_guardados) < 14:
            print(f"Siguiente: {NOMBRES_PUNTOS[len(puntos_guardados)]}")
        else:
            print("Completado. Pulsa cualquier tecla para salir.")

def click_general(event, x, y, flags, param):
    # Recorta una zona alrededor del clic para hacerle zoom
    global img_original
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntos_guardados) >= 14: return

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
        print(f"Error: No se encontró {RUTA_IMAGEN}")
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

    if len(puntos_guardados) == 14:
        ruta_salida_img = os.path.join(CARPETA, 'pista2D_marcados.png')
        ruta_salida_txt = os.path.join(CARPETA, 'puntos_referencia.txt')
        
        cv2.imwrite(ruta_salida_img, img_display)
        
        with open(ruta_salida_txt, 'w') as f:
            f.write("# Coordenadas para la clase CourtModel\n")
            f.write(f"self.p1_top_left_doubles   = {puntos_guardados[0]}\n")
            f.write(f"self.p2_top_right_doubles  = {puntos_guardados[1]}\n")
            f.write(f"self.p3_top_left_singles   = {puntos_guardados[2]}\n")
            f.write(f"self.p4_top_right_singles  = {puntos_guardados[3]}\n")
            f.write(f"self.p5_service_top_left   = {puntos_guardados[4]}\n")
            f.write(f"self.p6_service_top_right  = {puntos_guardados[5]}\n")
            f.write(f"self.p7_service_btm_left   = {puntos_guardados[6]}\n")
            f.write(f"self.p8_service_btm_right  = {puntos_guardados[7]}\n")
            f.write(f"self.p9_btm_left_singles   = {puntos_guardados[8]}\n")
            f.write(f"self.p10_btm_right_singles = {puntos_guardados[9]}\n")
            f.write(f"self.p11_btm_left_doubles  = {puntos_guardados[10]}\n")
            f.write(f"self.p12_btm_right_doubles = {puntos_guardados[11]}\n")
            f.write(f"self.top_t_point           = {puntos_guardados[12]}\n")
            f.write(f"self.bottom_t_point        = {puntos_guardados[13]}\n")
            
        print(f"Guardado en {CARPETA}")
    else:
        print("Operación cancelada. Faltan puntos por marcar.")

if __name__ == "__main__":
    main()