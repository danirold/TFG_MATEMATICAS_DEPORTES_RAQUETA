import cv2
import numpy as np
import os

CARPETA = 'dataset'
NOMBRE_IMAGEN = 'pista2D.png'
RUTA_IMAGEN = os.path.join(CARPETA, NOMBRE_IMAGEN)

FACTOR_ZOOM = 10     
TAMANO_LUPA = 60     

NOMBRES_PUNTOS = [
    "1. L1 (Fondo Sup): Exterior Izq", "2. L1 (Fondo Sup): Interior Izq", 
    "3. L1 (Fondo Sup): Centro", "4. L1 (Fondo Sup): Interior Der", "5. L1 (Fondo Sup): Exterior Der",
    "6. L2 (Saque Largo Sup): Exterior Izq", "7. L2 (Saque Largo Sup): Interior Izq", 
    "8. L2 (Saque Largo Sup): Centro", "9. L2 (Saque Largo Sup): Interior Der", "10. L2 (Saque Largo Sup): Exterior Der",
    "11. L3 (Saque Corto Sup): Exterior Izq", "12. L3 (Saque Corto Sup): Interior Izq", 
    "13. L3 (Saque Corto Sup): Centro", "14. L3 (Saque Corto Sup): Interior Der", "15. L3 (Saque Corto Sup): Exterior Der",
    "16. L4 (Saque Corto Inf): Exterior Izq", "17. L4 (Saque Corto Inf): Interior Izq", 
    "18. L4 (Saque Corto Inf): Centro", "19. L4 (Saque Corto Inf): Interior Der", "20. L4 (Saque Corto Inf): Exterior Der",
    "21. L5 (Saque Largo Inf): Exterior Izq", "22. L5 (Saque Largo Inf): Interior Izq", 
    "23. L5 (Saque Largo Inf): Centro", "24. L5 (Saque Largo Inf): Interior Der", "25. L5 (Saque Largo Inf): Exterior Der",
    "26. L6 (Fondo Inf): Exterior Izq", "27. L6 (Fondo Inf): Interior Izq", 
    "28. L6 (Fondo Inf): Centro", "29. L6 (Fondo Inf): Interior Der", "30. L6 (Fondo Inf): Exterior Der"
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
    cv2.resizeWindow('Mapa General', 500, 900)
    
    cv2.imshow('Mapa General', img_display)
    cv2.setMouseCallback('Mapa General', click_general)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(puntos_guardados) == TOTAL_PUNTOS:
        ruta_salida_img = os.path.join(CARPETA, 'Badminton_marcados.png')
        ruta_salida_txt = os.path.join(CARPETA, 'puntos_referencia_badminton.txt')
        
        cv2.imwrite(ruta_salida_img, img_display)
        
        with open(ruta_salida_txt, 'w', encoding='utf-8') as f:
            f.write("self.coordenadas_badminton = {\n")
            lineas = ["L1_fondo_sup", "L2_largo_sup", "L3_corto_sup", "L4_corto_inf", "L5_largo_inf", "L6_fondo_inf"]
            cols = ["ext_izq", "int_izq", "centro", "int_der", "ext_der"]
            
            idx = 0
            for l_name in lineas:
                for c_name in cols:
                    f.write(f"    '{l_name}_{c_name}': {puntos_guardados[idx]},\n")
                    idx += 1
            f.write("}\n")
            
        print(f"Guardado en {CARPETA}")
    else:
        print("Operación cancelada. Faltan puntos por marcar.")

if __name__ == "__main__":
    main()