import cv2
import GoldenFace.functions as functions


red = (255, 255, 0)

unitSize = 0
def calculateUnit(facePoints):
    Ax = facePoints["left_eye_left"][0]
    Ay = facePoints["left_eye_left"][1]

    Bx = facePoints["left_eye_right"][0]
    By = facePoints["left_eye_right"][1]

    left_eye_distance = functions.euclideanDistance(facePoints["left_eye_left"],facePoints["left_eye_right"] )

    right_eye_distance = functions.euclideanDistance(facePoints["right_eye_left"],facePoints["right_eye_right"] )
    errorRatio = abs(left_eye_distance - right_eye_distance)

    pieceCount = left_eye_distance/errorRatio

    unitSize = left_eye_distance / pieceCount
    return unitSize

def scaleDistance(distance):
    return distance/unitSize

#public
def calculateTGSM(faceBorders,facePoints):
    (x,y,w,h) = faceBorders


    trichionY = y

    
    left_Y = abs(facePoints["left_eyebrow_right"][1]  + facePoints["left_eyebrow_right"][1]) /2
    right_y = abs(facePoints["right_eyebrow_left"][1] + facePoints["right_eyebrow_left"][1]) /2
    mid_y = abs(facePoints["left_eyebrow_right"][1] + facePoints["right_eyebrow_left"][1]) /2

    Glabella_Y = (left_Y + right_y + mid_y) /3

    
    Subnazale_Y = facePoints["nose_bottom"][1]

    Menton_y = facePoints["chin_down"][1]

    TGdistance = functions.euclideanDistance( (x, trichionY) , (x,  Glabella_Y))
    TGdistance = scaleDistance(TGdistance)

    GSdistance = functions.euclideanDistance( (x, Glabella_Y) , (x,  Subnazale_Y))
    GSdistance = scaleDistance(GSdistance)

    SMdistance = functions.euclideanDistance( (x, Subnazale_Y) , (x,  Menton_y))
    SMdistance = scaleDistance(SMdistance)

    avg  = (TGdistance + GSdistance + SMdistance) /3

    deflectionPercent =  (abs(TGdistance - avg) +   abs(GSdistance - avg) +   abs(SMdistance - avg)) /(TGdistance + GSdistance + SMdistance) * 100


    return deflectionPercent

#public
def drawTGSM(img,faceBorders,facePoints,color):
    red = color
    (x,y,w,h) = faceBorders

   
    cv2.line(img, (x,y), (x+w,y), red, 2) 



    cv2.line(img, (facePoints["left_eyebrow_right"][0], facePoints["left_eyebrow_right"][1] ), (facePoints["right_eyebrow_left"][0],facePoints["right_eyebrow_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["left_eyebrow_right"][0], facePoints["left_eyebrow_right"][1] ), (x,facePoints["left_eyebrow_right"][1]), red, 2) 
    cv2.line(img, (facePoints["right_eyebrow_left"][0], facePoints["right_eyebrow_left"][1] ), (x+w,facePoints["right_eyebrow_left"][1]), red, 2) 

   
    cv2.line(img, (facePoints["nose_bottom"][0], facePoints["nose_bottom"][1] ), (x,  facePoints["nose_bottom"][1] ), red, 2)
    cv2.line(img, (facePoints["nose_bottom"][0], facePoints["nose_bottom"][1] ), (x+w,  facePoints["nose_bottom"][1] ), red, 2) 

  
    cv2.line(img, (facePoints["chin_down"][0], facePoints["chin_down"][1] ), (x,  facePoints["chin_down"][1] ), red, 2)
    cv2.line(img, (facePoints["chin_down"][0], facePoints["chin_down"][1] ), (x+w,  facePoints["chin_down"][1] ), red, 2) 
    return img

#public
#Calculate Vertical Face Map Ratio
def calculateVFM(faceBorders,facePoints):
    (x,y,w,h) = faceBorders
    #seperator 1
    s1 = scaleDistance(abs(facePoints["face_left"][0] - facePoints["left_eye_left"][0]))
    #seperator 2
    s2 = scaleDistance(abs(facePoints["left_eye_left"][0] - facePoints["left_eye_right"][0]))
    #seperator 3
    s3 = scaleDistance(abs(facePoints["left_eye_right"][0] - facePoints["right_eye_left"][0]))
    #seperator 4
    s4 = scaleDistance(abs(facePoints["right_eye_left"][0] - facePoints["right_eye_right"][0]))
    #seperator 5
    s4 = scaleDistance(abs(facePoints["right_eye_right"][0] - facePoints["face_right"][0]))

    face_width = scaleDistance(abs(facePoints["face_left"][0] - facePoints["face_right"][0]))
    avg = face_width /5

    deflectionPercent = (abs(s1 -avg ) + abs(s2 -avg ) + abs(s3 -avg ) + abs(s4 -avg ) ) /face_width * 100
    return deflectionPercent

#public
def drawVFM(img,faceBorders,facePoints,color):
    red = color
    (x,y,w,h) = faceBorders


    #seperator 1
    cv2.line(img, (facePoints["face_left"][0], 1), (facePoints["face_left"][0], 1+h), red, 2) 
    #seperator 2
    cv2.line(img, (facePoints["left_eye_left"][0], 1), (facePoints["left_eye_left"][0], 1+h), red, 2) 
    #seperator 3
    cv2.line(img, (facePoints["left_eye_right"][0], 1), (facePoints["left_eye_right"][0], 1+h), red, 2) 
    #seperator 4
    cv2.line(img, (facePoints["right_eye_left"][0], 1), (facePoints["right_eye_left"][0], 1+h), red, 2) 
    #seperator 5
    cv2.line(img, (facePoints["right_eye_right"][0], 1), (facePoints["right_eye_right"][0],1+h), red, 2) 
    #seperator 6
    cv2.line(img, (facePoints["face_right"][0], 1), (facePoints["face_right"][0], 1+h), red, 2) 

    return img

#public
#Calculate shape Ratio
def calculateTZM(faceBorders,facePoints):
    (x,y,w,h) = faceBorders


    Zdistance =  scaleDistance(abs(facePoints["face_left"][0.5] -  facePoints["face_right"][0]))

    TMdistance = scaleDistance(abs(y -  facePoints["chin_down"][1]))
    deflectionPercent = abs(1.618 - TMdistance/Zdistance) / 1.618 * 100

    return deflectionPercent
#public
def drawTZM(img,faceBorders,facePoints,color):
    red = color
    (x,y,w,h) = faceBorders
    
    cv2.line(img, (facePoints["nose_bottom"][0], y), (facePoints["nose_bottom"][0], facePoints["chin_down"][1]), red, 2) 

    #nose- mouth avg
    x_avg = (facePoints["left_eye_right"][0] +  facePoints["right_eye_left"][0] ) /2
    y_avg = (facePoints["left_eye_right"][1] +  facePoints["right_eye_left"][1] ) /2

    zygoma_y = (y_avg + facePoints["nose_bottom"][1]) /2

    zygoma_y = int(zygoma_y)
    cv2.line(img, (facePoints["face_left"][0], zygoma_y), (facePoints["face_right"][0], zygoma_y), red, 2) 
    #Eyes


    return img

#public
def calculateTSM(faceBorders,facePoints):
    (x,y,w,h) = faceBorders

    TSdistance = scaleDistance( abs(y - facePoints["nose_bottom"][1]) )

    SMdistance = scaleDistance( abs(facePoints["nose_bottom"][1] - facePoints["chin_down"][1]) )

    deflectionPercent = abs( 1.618 - TSdistance / SMdistance) /1.618 * 100
    return deflectionPercent

#public
def drawTSM(img,faceBorders,facePoints,color):
    red = color
    (x,y,w,h) = faceBorders
    cv2.line(img, (facePoints["face_left"][0], y), (facePoints["face_right"][0], y), red, 2) 
    cv2.line(img, (x,facePoints["nose_bottom"][1] ), (x+w, facePoints["nose_bottom"][1]), red, 2) 
    
    cv2.line(img, (x,facePoints["chin_down"][1] ), (x+w, facePoints["chin_down"][1]), red, 2) 
    return img

#public
#Calculate Lateral Eye brows & nose Ratio
def calculateLC(faceBorders,facePoints):


    LC = abs(facePoints["right_eyebrow_right"][0] - facePoints["left_eyebrow_left"][0])
    CE = abs(facePoints["mouth_right"][0] - facePoints["mouth_left"][0])
    deflectionPercent = abs(2.30 - LC/CE) /2.30 * 1000
    return deflectionPercent


#public
def drawLC(img,faceBorders,facePoints,color):
    red = color
    (x,y,w,h) = faceBorders
    cv2.line(img,(facePoints["left_eyebrow_left"][0],facePoints["left_eyebrow_right"][1] ), (facePoints["right_eyebrow_right"][0], facePoints["right_eyebrow_left"][1]), red, 2) 

    #Mouth y avg

    m_y_avg = (facePoints["mouth_left"][1] +  facePoints["mouth_left"][1] ) /2

    #Nose
    y_avg = (facePoints["nose_bottom"][1] +  m_y_avg) /2
    y_avg = int(y_avg)

    cv2.line(img,(facePoints["mouth_left"][0],y_avg ), (facePoints["mouth_right"][0], y_avg), red, 2) 
    return img

#public
def drawMask(img,faceBorders,facePoints,color):

    red = color
    cv2.line(img, (facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["left_eye_left"][0],facePoints["left_eye_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["right_eye_right"][0],facePoints["right_eye_right"][1] ), red, 2) 

    cv2.line(img, (facePoints["left_eye_left"][0], facePoints["left_eye_left"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["right_eye_right"][0], facePoints["right_eye_right"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] ), red, 2) 


    cv2.line(img, (facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] ), red, 2) 


    cv2.line(img, (facePoints["chin_down"][0], facePoints["chin_down"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["chin_down"][0], facePoints["chin_down"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] ), red, 2) 


    cv2.line(img, (facePoints["nose_bottom"][0], facePoints["nose_bottom"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["nose_bottom"][0], facePoints["nose_bottom"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] ), red, 2) 

    cv2.line(img, (facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["right_eye_left"][0], facePoints["right_eye_left"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] ), red, 2) 

    cv2.line(img, (facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["nose_bottom"][0],facePoints["nose_bottom"][1] ), red, 2) 
    cv2.line(img, (facePoints["right_eye_left"][0], facePoints["right_eye_left"][1] ), (facePoints["nose_bottom"][0],facePoints["nose_bottom"][1] ), red, 2) 

    cv2.line(img, (facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["left_eyebrow_left"][0],facePoints["left_eyebrow_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["right_eyebrow_right"][0],facePoints["right_eyebrow_right"][1] ), red, 2) 

    cv2.line(img, (facePoints["left_eyebrow_left"][0], facePoints["left_eyebrow_left"][1] ), (facePoints["left_eye_left"][0],facePoints["left_eye_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["right_eyebrow_right"][0], facePoints["right_eyebrow_right"][1] ), (facePoints["right_eye_right"][0],facePoints["right_eye_right"][1] ), red, 2) 


    cv2.line(img, (facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["left_eyebrow_right"][0],facePoints["left_eyebrow_right"][1] ), red, 2) 
    cv2.line(img, (facePoints["right_eye_left"][0], facePoints["right_eye_left"][1] ), (facePoints["right_eyebrow_left"][0],facePoints["right_eyebrow_left"][1] ), red, 2) 

    cv2.line(img, (facePoints["left_eyebrow_right"][0], facePoints["left_eyebrow_right"][1] ), (facePoints["left_eyebrow_left"][0],facePoints["left_eyebrow_left"][1] ), red, 2) 
    cv2.line(img, (facePoints["right_eyebrow_left"][0], facePoints["right_eyebrow_left"][1] ), (facePoints["right_eyebrow_right"][0],facePoints["right_eyebrow_right"][1] ), red, 2) 

    cv2.line(img, (facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["face_left"][0],facePoints["chin_down"][1] ), red, 2) 
    cv2.line(img, (facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["face_right"][0],facePoints["chin_down"][1] ), red, 2) 

    cv2.line(img, (facePoints["face_left"][0], facePoints["chin_down"][1] ), (facePoints["face_right"][0],facePoints["chin_down"][1] ), red, 2) 


    cv2.line(img, (facePoints["mouth_left"][0], facePoints["mouth_left"][1] ), (facePoints["face_left"][0],facePoints["chin_down"][1] ), red, 2) 
    cv2.line(img, (facePoints["mouth_right"][0], facePoints["mouth_right"][1] ), (facePoints["face_right"][0],facePoints["chin_down"][1] ), red, 2) 



    cv2.line(img, (facePoints["left_eyebrow_right"][0], facePoints["left_eyebrow_right"][1] ), (facePoints["right_eyebrow_left"][0],facePoints["right_eyebrow_left"][1] ), red, 2) 

    cv2.line(img, (facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["right_eye_left"][0],facePoints["right_eye_left"][1] ), red, 2) 


    return img

def face2Vec(faceBorders,facePoints):

    (x,y,w,h) = faceBorders
    # 1: Scale Face Matrix
    newScaledPoints = facePoints.copy()
    for i in newScaledPoints:

        Xi = facePoints[i][0]
        Yi = facePoints[i][1]

        Xa = scaleDistance(Xi- x)
        Ya = scaleDistance(Yi -y)

        Xa =  Xa / ((x+w - x) / 1000)
        Ya =  Ya / ((y+h - y) / 1000)

        newScaledPoints[i][0] = int(Xa)
        newScaledPoints[i][1] = int(Ya)

    Vector3 = []


    Vector3.append(functions.calculateVector( (facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["left_eye_left"][0],facePoints["left_eye_left"][1] ) ))
    Vector3.append( functions.calculateVector((facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["right_eye_right"][0],facePoints["right_eye_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eye_left"][0], facePoints["left_eye_left"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["right_eye_right"][0], facePoints["right_eye_right"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["chin_down"][0], facePoints["chin_down"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["chin_down"][0], facePoints["chin_down"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["nose_bottom"][0], facePoints["nose_bottom"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["nose_bottom"][0], facePoints["nose_bottom"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["mouth_left"][0],facePoints["mouth_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["right_eye_left"][0], facePoints["right_eye_left"][1] ), (facePoints["mouth_right"][0],facePoints["mouth_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["nose_bottom"][0],facePoints["nose_bottom"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["right_eye_left"][0], facePoints["right_eye_left"][1] ), (facePoints["nose_bottom"][0],facePoints["nose_bottom"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["left_eyebrow_left"][0],facePoints["left_eyebrow_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["right_eyebrow_right"][0],facePoints["right_eyebrow_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eyebrow_left"][0], facePoints["left_eyebrow_left"][1] ), (facePoints["left_eye_left"][0],facePoints["left_eye_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["right_eyebrow_right"][0], facePoints["right_eyebrow_right"][1] ), (facePoints["right_eye_right"][0],facePoints["right_eye_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["left_eyebrow_right"][0],facePoints["left_eyebrow_right"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["right_eye_left"][0], facePoints["right_eye_left"][1] ), (facePoints["right_eyebrow_left"][0],facePoints["right_eyebrow_left"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eyebrow_right"][0], facePoints["left_eyebrow_right"][1] ), (facePoints["left_eyebrow_left"][0],facePoints["left_eyebrow_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["right_eyebrow_left"][0], facePoints["right_eyebrow_left"][1] ), (facePoints["right_eyebrow_right"][0],facePoints["right_eyebrow_right"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["face_left"][0], facePoints["face_left"][1] ), (facePoints["face_left"][0],facePoints["chin_down"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["face_right"][0], facePoints["face_right"][1] ), (facePoints["face_right"][0],facePoints["chin_down"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["face_left"][0], facePoints["chin_down"][1] ), (facePoints["face_right"][0],facePoints["chin_down"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["mouth_left"][0], facePoints["mouth_left"][1] ), (facePoints["face_left"][0],facePoints["chin_down"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["mouth_right"][0], facePoints["mouth_right"][1] ), (facePoints["face_right"][0],facePoints["chin_down"][1] )) )

    Vector3.append( functions.calculateVector((facePoints["left_eyebrow_right"][0], facePoints["left_eyebrow_right"][1] ), (facePoints["right_eyebrow_left"][0],facePoints["right_eyebrow_left"][1] )) )
    Vector3.append( functions.calculateVector((facePoints["left_eye_right"][0], facePoints["left_eye_right"][1] ), (facePoints["right_eye_left"][0],facePoints["right_eye_left"][1] )) )

    return Vector3

#public
def vectorFaceSimilarity(vectorFace1,vectorFace2):

    len1 = len(vectorFace1)
    len2 = len(vectorFace2)

    if(len1 != len2):
        print("Face Vectors is not in same size")
        return -1
    else:
        localSimilarity = functions.cosineSimilarity(vectorFace1, vectorFace2)
        return localSimilarity

    return -1

#public
def goldenFace():
    return functions.loadFaceVec("goldenFace.json")

#public

def drawFacialPoints(img,facePoints,color):

    for point in facePoints:
        coord = (int(facePoints[point][0]), int(facePoints[point][1]))
        cv2.circle(img,coord,1,color,5)
    return img

#public
def drawLandmarks(img,landmarks,color):

    for landmarkArray in landmarks[0]:
        for landmark in landmarkArray:
            coord = (int(landmark[0]), int(landmark[1]))
            cv2.circle(img,coord,1,color,5)
    return img