import cv2
import os

for chord in os.listdir('./videos'):
    currentFrame = 0

    for video in os.listdir('./videos/'+chord):        
        ret = True
        cap = cv2.VideoCapture(f'videos/' + chord + "/" + video)
        print(cap)
        if cap.isOpened():
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                
                if ret == False: break
                print("ret: ", ret)
                print("Frame: ", frame, "\n")

                # Saves image of the current frame in jpg file
                name = 'dataset/' + chord + "/frame_" + str(currentFrame) + '.jpg'
                print ('Creating...' + name)
                cv2.imwrite(name, frame)

                # To stop duplicate images
                currentFrame += 1
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()