import cv2
import imutils

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]  # Select 
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        print(f"Eye Coordinates: x={ex}, y={ey}, width={ew}, height={eh}")

    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=20, minSize=(40, 40))
    mouth = sorted(mouth, key=lambda x: x[2] * x[3], reverse=True)[:1]  # 
    
    for (mx, my, mw, mh) in mouth:
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)
        print(f"Mouth Coordinates: x={mx}, y={my}, width={mw}, height={mh}")

    img = imutils.resize(img, width=600)
    cv2.imshow('Image with Features', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_paths = ['./Images/00056.png', './Images/00059.png', './Images/00081.png', './Images/01025.png', './Images/09074.png', './Images/09126.png']
for image_path in image_paths:
    features = detect_features(image_path)
    print(features)