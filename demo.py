import sys
import os

try:
    import numpy as np
    import cv2
except ImportError and ModuleNotFoundError as e:
    print("OpenCV, Numpy is not Imported \n"
          "with :", e)
else: print("Import OpenCV\n"
            "Import Numpy")

try:
    assert sys.platform == 'win32'
    assert sys.version_info[:2] >= (3,6)
except AssertionError:
    msg = ("System is Supported in\n"
           "windows10\n"
           "python 3.6 or upper")
    raise AssertionError(msg)

if __name__=="__main__":
    cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("window width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("window height :", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    try:
        print("===========================================\n"
              "              Start Detection\n"
              "===========================================")
        while True:
            ret, frame = cap.read()
            
            if ret is True:
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                
                dst = cv2.cornerHarris(gray,2,3,0.04)
                dst = cv2.dilate(dst,None)
                
                frame[dst>0.01*dst.max()]=[0,0,255]
                
                # Edge derivative
                sobel = cv2.Sobel(gray.copy(), cv2.CV_8U, 1, 0, 3)
                laplacian = cv2.Laplacian(gray.copy(),cv2.CV_8U,ksize=5)
                canny = cv2.Canny(frame.copy(), 100, 255)                
                
                inf = np.hstack((sobel,canny))
                cv2.imshow('Object Labeling', canny)
                
                # Settings Object Annotation File Directory
                #path = os.path.join(os.getcwd(),"data_annotation_file")
                #if not os.path.isdir(path):
                #    os.mkdir(path)
                #else:
                #    pass
            
                # Image Feature Matching
                # 1. Color Histograms (Use Only Color Data)
                # 2. Corner Matching  (Use Corner Detector)
                # 3. Color Sptiograms (Use Color Data each area)
                # 4. (HOG)Histogram of Oriented Gradients (Not Use Color Data)
                
                # Explain
                # Object Data has not deformation problem
                # Object Data has Occulusion problem
                
            else:
                print("cap.read() Error")
                
            if cv2.waitKey(1):
                pass
    except KeyboardInterrupt:
        print("===========================================\n"
              "Release VideoCapture with KeyboardInterrupt\n"
              "===========================================")
        cap.release()
        cv2.destroyAllWindows()
