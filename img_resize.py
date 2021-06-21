
import cv2
import numpy as np

for i in range(8):
    img = cv2.imread("target_rsz/class_D_" + str(i*45) + ".jpg")
    img1 = cv2.Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                         -1, 5)
    
    kernel1 = np.ones((3,3))
    img2 = cv2.dilate(img1, kernel1, iterations = 1)
    
    cv2.imwrite("target_lpl_dil/class_D_" + str(i*45) + ".jpg",img2)
    
    

"""
def a(k):
    global q
    q = k
    while q < 111112000:
        q=q+1
        #print(q)
    
import threading

th=threading.Thread(target=a, args=(1000,))
th.daemon=True
th.start()

while True:
    print(q)
    """

"""
import cv2

cap = cv2.VideoCapture(0)
	
while True:
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.Canny(frame, 50, 200, apertureSize=3, L2gradient=True)
        cv2.imshow('video', frame)
    else:
        print('error')
        
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()	
"""