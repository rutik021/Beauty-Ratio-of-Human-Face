import cv2
import GoldenFace.goldenMath
import GoldenFace.functions



def detectLandmark(landmarks):

    points = []

    ### Facial Points ###
    face_left = []
    left_eye_left = []
    left_eye_right = []
    right_eye_left = []
    right_eye_right = []
    face_right = []

    mouth_left = []
    mouth_right = []

    left_eyebrow_left = []
    left_eyebrow_right = []

    right_eyebrow_left = []
    right_eyebrow_right = []


    nose_left = []
    nose_right = []
    nose_bottom = []


    chin_down = []


    color_main  = (0,0,255)
    
    for landmark in landmarks:
        array = landmark[0]
        i = 0
        for l in array:
            x = l[0]
            y = l[1]

            if i <= 15 and i>=0:

                if i==0:
                    face_left = [x,y]

                if i==8:
                    chin_down= [x,y]

            if i<=20 and i >16:

                if i== 17:
                    left_eyebrow_left = [x,y]

            if i==21:
                left_eyebrow_right = [x,y]

            if i<=25 and i >21:
                if i ==22:
                    right_eyebrow_left = [x,y]

            if i ==26:
                 right_eyebrow_right=  [x,y]


            if i<=34 and i >30:

                if i==31:
                     nose_left = [x,y]

                if i==33:
                     nose_bottom = [x,y]


            if i==35:
                nose_right = [x,y]

            if i<=40 and i >35:


                if i== 36:  #start left eye
                    left_eye_left = [x,y]

                if i== 39:  #end left eye
                    left_eye_right = [x,y]

            if i<=46 and i >41:

                if i== 42:  #start left eye
                    right_eye_left = [x,y]

                if i== 45:  #end left eye
                    right_eye_right = [x,y]


            if i==48:
                mouth_left = [x,y]

          
            if i<=59 and i >48:
                if i==54:
                    mouth_right = [x,y]
                
            if i==16:
                face_right = [x,y]

            i = i + 1


            if i == 68:
                break

    points = {
    "face_left":face_left,
    "face_right":face_right,
    "left_eye_left":left_eye_left,
    "left_eye_right":left_eye_right,
    "right_eye_left":right_eye_left,
    "right_eye_right":right_eye_right,
    "mouth_left":mouth_left,
    "mouth_right":mouth_right,
    "left_eyebrow_left":left_eyebrow_left,
    "left_eyebrow_right":left_eyebrow_right,
    "right_eyebrow_left":right_eyebrow_left,
    "right_eyebrow_right":right_eyebrow_right,
    "nose_left":nose_left,
    "nose_right":nose_right,
    "nose_bottom":nose_bottom,
    "chin_down":chin_down,
    "face_right":face_right
    }
    return points


def drawLandmark(img,landmarks,color):
    color_main  = color
    for landmark in landmarks:
        array = landmark[0]
        i = 0
        for l in array:
            x = l[0]
            y = l[1]

            if i <= 15 and i>=0:

                if i==0:
                    face_left = [x,y]

                if i==8:
                    chin_down= [x,y]
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]),color_main, 3)

            if i<=20 and i >16:

                if i== 17:
                    left_eyebrow_left = [x,y]
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 4)

            if i==21:
                left_eyebrow_right = [x,y]

            if i<=25 and i >21:
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 4)
                if i ==22:
                    right_eyebrow_left = [x,y]

            if i ==26:
                 right_eyebrow_right=  [x,y]

            if i<=29 and i >26:
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]),color_main, 6)

            if i<=34 and i >30:

                if i==31:
                     nose_left = [x,y]

                if i==33:
                     nose_bottom = [x,y]

                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 4)

            if i==35:
                nose_right = [x,y]

            if i<=40 and i >35:
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 2)


                if i== 36:  #start left eye
                    left_eye_left = [x,y]

                if i== 39:  #end left eye
                    left_eye_right = [x,y]

                if i == 40:
                    cv2.line(img, (array[36][0],array[36][1]),(array[41][0], array[41][1]), color_main, 2)

            if i<=46 and i >41:
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 2)


                if i== 42:  #start left eye
                    right_eye_left = [x,y]

                if i== 45:  #end left eye
                    right_eye_right = [x,y]

                if i == 46:
                    cv2.line(img, (x,y),(array[42][0], array[46][1]),color_main, 2)

            if i==48:
                mouth_left = [x,y]

            
            if i<=59 and i >48:
                if i==54:
                    mouth_right = [x,y]

                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 4)
                if i == 59:
                    cv2.line(img, (array[48][0],array[48][1]),(array[50][0], array[50][1]), color_main, 4)
         
            if i<=62 and i >60:
                cv2.line(img, (x,y),(array[i+1][0], array[i+1][1]), color_main, 4)

            if i==16:
                face_right = [x,y]

            i = i + 1


            if i == 68:
                break

    return img
