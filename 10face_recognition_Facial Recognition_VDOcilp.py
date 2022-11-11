import face_recognition as face
import numpy as np
import cv2


# open file if you change to 0 that means you open webcam instead
video_capture = cv2.VideoCapture("D:/Work-DRR/code/opencvtube/src/saved-media/windermere.mp4") 

# First create person object to reference
mickey_image = face.load_image_file("D:/Work-DRR/code/opencvtube/src/images/face_personal/mickey.jpg")
mickey_face_encoding = face.face_encodings(mickey_image)[0] # it is expend matrix 128 ea 

# Second create person object to reference
ken_image = face.load_image_file("D:/Work-DRR/code/opencvtube/src/images/face_personal/ken.jpg")
ken_face_encoding = face.face_encodings(ken_image)[0] # it is expend matrix 128 ea 

# mary_image = face.load_image_file("mary.jpg")
# mary_face_encoding = face.face_encodings(mary_image)[0] # it is expend matrix 128 ea 

# sun_image = face.load_image_file("sun.jpg")
# sun_face_encoding = face.face_encodings(sun_image)[0] # it is expend matrix 128 ea 

# pee_image = face.load_image_file("pee.jpg")
# pee_face_encoding = face.face_encodings(pee_image)[0] # it is expend matrix 128 ea 

# iris_image = face.load_image_file("iris.jpg")
# iris_face_encoding = face.face_encodings(iris_image)[0] # it is expend matrix 128 ea 
# ### ------- Set original face encoding dataset -------------- ###

# Variable
face_localtions = [] # location of picture face (list)
face_encodings = [] # feature encoding
face_names = [] # name of face
face_percent = [] # percentage of sure
process_this_frame =  True # process to calculating to be increased in fps

known_face_encodings = [mickey_face_encoding, ken_face_encoding] # input face to check reference known or not
known_face_names = ["mickey", "ken"]

# loop calculating each frame of Video
while True:
    ret, frame = video_capture.read() # ret is read from true or false || frame is picture of loop in VDO
    if ret:
        
        # inpuyr data by reduce the size of picture /2 and then when show the data *2 for reduce time to process_this_frame
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
         # make to be RGB for calculating
        rgb_small_frame = small_frame[:,:,::-1] # [weidge, high, RGB to BGR]
        
        face_names = []
        face_percent = []
        
        if process_this_frame: # check process_this_frame for extracting time to check True or False
            face_localtions = face.face_locations(rgb_small_frame, model="cnn") # if true
            face_encodings = face.face_encodings(rgb_small_frame, face_localtions) #
            
            # compare each face to determine
            for face_encoding in face_encodings:
                # check face encoding to same of object if the value is high that means not same but the value is low that it is same
                face_distances = face.face_distance(known_face_encodings, face_encoding) #)
                best_match = np.argmin(face_distances) #returb the lowest distance
                face_percent_value = 1 - face_distances[best_match] #]
                
                # filter face_percent_value person
                if face_percent_value >= 0.5: # if more than 50%, the data will be show on the video
                    name = known_face_names[best_match]
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                else:
                    name = "UNKNOWN"
                    face_percent.append(0)
                face_names.append(name)
                
        # draw the square to show the best match      
        for (top,right,bottom,left), name, percent in zip(face_localtions, face_names, face_percent):      # zip is a catch the data 
            top*=2 # up size *2
            right*=2 # up size *2
            bottom*=2 # up size *2
            left*=2 # up size *2
            
            if name == "UNKNOWN":
                color = [46,2,209] # red color
            else:
                color = [255,102,51] # blue color
                
            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame, (left-1,top -30), (right+1,top), color, cv2.FILLED)
            cv2.rectangle(frame, (left-1,bottom), (right+1,bottom+30), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, top-6), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, "MATCH:  "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (255,255,255), 1)

        # change the value of the frame to calculating every frame
        process_this_frame = not process_this_frame
        
        # get the result
        cv2.imshow("Video", frame) # show video in loop in frame
        if cv2.waitKey(1) & 0xFF == ord('q'): # meaning how long do you want to show (miliseconds)
            break
        
    else:
        break
    
# clear value when close window
video_capture.release()
cv2.destroyAllWindows()