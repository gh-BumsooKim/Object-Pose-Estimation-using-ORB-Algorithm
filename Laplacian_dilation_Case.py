import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

orb = cv2.ORB_create(nfeatures = 1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

img_target_a = list()
img_target_b = list()
img_target_c = list()
img_target_d = list()

kernel = np.ones((3,3))

for i in range(8):
    temp = cv2.imread("target_rsz/class_A_" + str(i*45) + ".jpg")
    img_target_a.append(cv2.dilate(cv2.Laplacian(
        cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),-1, 5), kernel, iterations = 1))

for i in range(8):
    temp = cv2.imread("target_rsz/class_B_" + str(i*45) + ".jpg")
    img_target_b.append(cv2.dilate(cv2.Laplacian(
        cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),-1, 5), kernel, iterations = 1))
    
for i in range(8):  
    temp = cv2.imread("target_rsz/class_C_" + str(i*45) + ".jpg")
    img_target_c.append(cv2.dilate(cv2.Laplacian(
        cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),-1, 5), kernel, iterations = 1))

for i in range(8):
    temp = cv2.imread("target_rsz/class_D_" + str(i*45) + ".jpg")
    img_target_d.append(cv2.dilate(cv2.Laplacian(
        cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),-1, 5), kernel, iterations = 1))

img_target_list = [img_target_a[0], img_target_b[0],
                   img_target_c[0], img_target_d[0]]
img_target = img_target_a[0]

adp_dtc = list()

#def pointMatching(des, adt_idx):

STEP = 1
TRG = 0

while True:
    ret, frame = cap.read()
    
    if ret == True:       
        
        #frame = cv2.Canny(frame, 100, 200, apertureSize=3, L2gradient=True)
        frame = cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),-1, 5)
        frame = cv2.dilate(frame, kernel, iterations = 1)
        kp0, des0 = orb.detectAndCompute(frame, None)
        
        # Classifier
        if STEP == 1:
            kp1, des1 = orb.detectAndCompute(img_target_a[0], None)
            matchesA = bf.match(des0, des1)
            matchesA = sorted(matchesA, key = lambda x:x.distance)
            
            kp2, des2 = orb.detectAndCompute(img_target_b[0], None)
            matchesB = bf.match(des0, des2)
            matchesB = sorted(matchesB, key = lambda x:x.distance)
            
            kp3, des3 = orb.detectAndCompute(img_target_c[0], None)
            matchesC = bf.match(des0, des3)
            matchesC = sorted(matchesC, key = lambda x:x.distance)
            
            kp4, des4 = orb.detectAndCompute(img_target_d[0], None)
            matchesD = bf.match(des0, des4)
            matchesD = sorted(matchesD, key = lambda x:x.distance)
            
            matches_list = [matchesA, matchesB, matchesC, matchesD]
            kp_list = [kp1, kp2, kp3, kp4]
            
            max_matches = np.argmax([len(matchesA),len(matchesB),
                                     len(matchesC),len(matchesD)])
        
        # Topology Awareness
        # 0 -> A | 1 -> B
        # 2 -> C | 3 -> D
        if STEP == 2:
            if TRG == 0:
                matchesA = [None] * 8
                matchesA_arg = [None] * 8
                kp1 = [None] * 8
                des1 = [None] * 8
                
                for i in range(8):
                    kp1[i], des1[i] = orb.detectAndCompute(img_target_a[i], None)
                    matchesA[i] = bf.match(des0, des1[i])
                    matchesA[i] = sorted(matchesA[i], key = lambda x:x.distance)
                    matchesA_arg[i] = len(matchesA[i])
                    
                matches_list = matchesA
                img_target_list = img_target_a
                kp_list = kp1                
                max_matches = np.argmax(matchesA_arg)
                    
            if TRG == 1:
                matchesB = [None] * 8
                matchesB_arg = [None] * 8
                kp2 = [None] * 8
                des2 = [None] * 8
                
                for i in range(8):
                    kp2[i], des2[i] = orb.detectAndCompute(img_target_b[i], None)
                    matchesB[i] = bf.match(des0, des2[i])
                    matchesB[i] = sorted(matchesB[i], key = lambda x:x.distance)
                    matchesB_arg[i] = len(matchesB[i])
                    
                matches_list = matchesB
                img_target_list = img_target_b
                kp_list = kp2
                max_matches = np.argmax(matchesB_arg)
                
            if TRG == 2:
                matchesC = [None] * 8
                matchesC_arg = [None] * 8
                kp3 = [None] * 8
                des3 = [None] * 8
                
                for i in range(8):
                    kp3[i], des3[i] = orb.detectAndCompute(img_target_c[i], None)
                    matchesC[i] = bf.match(des0, des3[i])
                    matchesC[i] = sorted(matchesC[i], key = lambda x:x.distance)
                    matchesC_arg[i] = len(matchesC[i])
                    
                matches_list = matchesC
                img_target_list = img_target_c
                kp_list = kp3
                max_matches = np.argmax(matchesC_arg)
                
            if TRG == 3:
                matchesD = [None] * 8
                matchesD_arg = [None] * 8
                kp4 = [None] * 8
                des4 = [None] * 8
                
                for i in range(8):
                    kp4[i], des4[i] = orb.detectAndCompute(img_target_d[i], None)
                    matchesD[i] = bf.match(des0, des4[i])
                    matchesD[i] = sorted(matchesD[i], key = lambda x:x.distance)
                    matchesD_arg[i] = len(matchesD[i])
                    
                matches_list = matchesD
                img_target_list = img_target_d
                kp_list = kp4
                max_matches = np.argmax(matchesD_arg)
                
            print("Rererere")
        
        # Adaptive Detection - LRU(Least Recently used)
        if len(adp_dtc) >= 60:
            adp_dtc.pop(0)
            adp_dtc.append(max_matches)
        else:
            adp_dtc.append(max_matches)
        
        adp_idx = np.argmax(np.bincount(adp_dtc))
        print(adp_idx)
        
        if list(filter(lambda x:x>30, np.bincount(adp_dtc))) and STEP == 1:
            #STEP = 2
            TRG = adp_idx
        
        # Max Argument
        matches = matches_list[adp_idx]
        img_target = img_target_list[adp_idx]
        kp = kp_list[adp_idx]
        
        flag = (cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS |
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Visualizing
        count = 80
        frame = cv2.drawMatches(frame, kp0, img_target, kp, 
                                matches[:count], None, flags=flag)
        
        cv2.imshow("Pre-Processing", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        print("Not Connected Camera")
        break
    
cap.release()
cv2.destroyAllWindows()
print("Cap released")

