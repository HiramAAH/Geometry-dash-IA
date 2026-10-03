import mss
import numpy as np
import cv2
import time #count frames

region = {#def the position and area
    'top' : 30,#y
    'left' : 800,#x
    'width' : 640,#x
    'height' : 480#y
}

with mss.MSS() as sct:
    last_frame = time.perf_counter()
    while True:
        #calculate frames per second
        now = time.perf_counter()#get the now time
        delta_time = now - last_frame
        last_frame = now
        #cap the frame
        screenshot = sct.grab(region)

        #change to OpenCV image 
        frame = np.array(screenshot)

        #convert to binary (esta medio roto porque no detecta las bases paralelas al suelo)
        _, binary_frame=cv2.threshold(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60)),117,255,cv2.THRESH_BINARY)
        #binary_frame = cv2.Canny(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60)), 100, 200)
        #resize the binary frame for display
        display_frame = cv2.resize(binary_frame, (640, 480), interpolation=cv2.INTER_NEAREST)

        # 1. Encontrar las formas blancas en la imagen binaria
        # RETR_EXTERNAL toma solo los bordes exteriores, CHAIN_APPROX_SIMPLE optimiza la memoria
        contornos, jerarquia = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear una versión a color del frame redimensionado para poder dibujar líneas de colores
        resized_color = cv2.resize(frame, (80, 60))

        # 2. Iterar sobre cada forma encontrada
        for contorno in contornos:
            # Ignorar figuras minúsculas (ruido visual)
            if cv2.contourArea(contorno) < 10:
                continue

            # 1. Calcular el perímetro de la figura
            perimetro = cv2.arcLength(contorno, True)
            
            # 2. Aproximar la forma geométrica
            # El 0.04 es un factor de precisión. Si la figura es irregular, la fuerza a ser un polígono simple.
            aproximacion = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
            
            # 3. Obtener las coordenadas de la hitbox
            x, y, w, h = cv2.boundingRect(aproximacion)
            
            # 4. Clasificar según el número de vértices (esquinas)
            if len(aproximacion) == 3:
                # Tiene 3 esquinas: Es un PINCHO. Dibujar hitbox ROJA.
                cv2.rectangle(resized_color, (x, y), (x + w, y + h), (0, 0, 255), 1)
                
            elif len(aproximacion) == 4:
                # Tiene 4 esquinas: Es un BLOQUE. Dibujar hitbox AZUL.
                cv2.rectangle(resized_color, (x, y), (x + w, y + h), (255, 0, 0), 1)
                
            else:
                # Otra forma no identificada. Dibujar hitbox VERDE.
                cv2.rectangle(resized_color, (x, y), (x + w, y + h), (0, 255, 0), 1)

        
        #show the frames
        cv2.imshow("Vision de la IA (Ampliada)", display_frame)
        cv2.imshow("Visión con Hitboxes", cv2.resize(resized_color, (640, 480), interpolation=cv2.INTER_NEAREST))

        print(int(1/delta_time))
        if cv2.waitKey(1) == ord('q'):
            break
        

cv2.destroyAllWindows()
