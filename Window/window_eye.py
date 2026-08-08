import mss
import numpy as np
import cv2
import time #count frames

region = {#def the position and area
    'top' : 250,#y
    'left' : 560,#x
    'width' : 800,#x
    'height' : 600#y
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        #show
        cv2.imshow("Window Game", frame )

        print(int(1/delta_time))
        if cv2.waitKey(1) == ord('q'):
            break
        

cv2.destroyAllWindows()
