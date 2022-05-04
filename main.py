# using haar cascade

import cv2 as cv

img = cv.imread("/root/Pictures/group 2.jpg")
# cv.imshow("face", img)

rsize = cv.resize(img,(800,500),interpolation=cv.INTER_CUBIC)
# cv.imshow('resize', rsize)

"""
cascade uses edges to determine mage colour doesnt matter
"""

gray = cv.cvtColor(rsize , cv.COLOR_BGR2GRAY)
cv.imshow("gray" , gray)

# reading the xml file
cascade = cv.CascadeClassifier('face.xml')

# detect face in image
faces_rectangle = cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

print(f"no of faces found = {len(faces_rectangle)}")

# drawing a rectangle on the face detected

for (x,y,w,h) in faces_rectangle:
    cv.rectangle(rsize, (x,y), (x+w,y+h), (0,255,0), thickness=2 )

cv.imshow("face detect", rsize)




cv.waitKey(0)
