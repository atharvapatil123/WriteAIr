import cv2
import numpy as np
import time


# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# This variable determines if we want to load color range from memory or use the ones defined here. 
load_from_disk = True

# If true then load color range from memory
if load_from_disk:
    penval = np.load('penval2.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Load these 2 images and resize them to the same size.
pen_img = cv2.resize(cv2.imread('pen.png',1), (75, 75))
eraser_img = cv2.resize(cv2.imread('eraser.jpg',1), (75, 75))

# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# Creating A 5x5 kernel for morphological operations
kernel = np.ones((5,5),np.uint8)

# Initializing the canvas on which we will draw upon
canvas = None

# Create a background subtractor Object
backgroundobject = cv2.createBackgroundSubtractorMOG2( detectShadows = False )

# This threshold determines the amount of disruption in background.
background_threshold = 600

# A variable which tells you if you're using a pen or an eraser.
switch = 'Pen'

# With this variable we will monitor the time between previous switch.
last_switch = time.time()

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800

# Threshold for wiper, the size of the contour must be bigger than for us to clear the canvas
wiper_thresh = 40000

# A varaible which tells when to clear canvas, if its True then we clear the canvas
clear = False

# This threshold is used to filter noise, the contour area must be bigger than this to qualify as an actual contour.
# noiseth = 500

while(1):
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip( frame, 1 )

    # Initilize the canvas as a black image of same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame) + 255
        # canvas = np.zeros_like(frame,(471,636,3)) + 255   

    # Take the top left of the frame and apply the background subtractor there    
    top_left = frame[0: 75, 0: 75]
    fgmask = backgroundobject.apply(top_left)
    # print(fgmask)
    # Note the number of pixels that are white, this is the level of disruption.
    switch_thresh = np.sum(fgmask==255)

    # If the disruption is greater than background threshold and there has been some time after the previous switch then you 
    # can change the object type.
    if switch_thresh > background_threshold  and (time.time() - last_switch) > 1:
        
        # Save the time of the switch. 
        last_switch = time.time()
        
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'


    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges from there
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
        # print(lower_range, upper_range)
        
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([26,80,147])
       upper_range = np.array([81,255,255])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Perform the morphological operations to get rid of the noise.
    # Erosion Eats away the white part while dilation expands it.
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)

    # res = cv2.bitwise_and(frame,frame, mask= mask)

    # mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # stack all frames and show it
    # stacked = np.hstack((mask_3,frame,res))
    # cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
        # Find Contours in the frame.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also make sure its size is bigger than noise threshold.
    if contours and cv2.contourArea(max(contours, 
                                        key = cv2.contourArea)) > noiseth:
        
        # Grab the biggest contour with respect to area
        c = max(contours, key = cv2.contourArea)
        
        # cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        # ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        # cv2.circle(canvas, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        # Get bounding box coordinates around that contour
        x2,y2,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x2,y2),(x2+w,y2+h),(0,25,255),2)      
        
        # Get the area of the contour
        area = cv2.contourArea(c)

        # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
        # This is true when we writing for the first time or when writing again when the pen had disapeared from view.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2

        else:
            if switch == 'Pen':
                # Draw the line on the canvas
                canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,0,0], 5)
                # ((x, y), radius) = cv2.minEnclosingCircle(c)
                
            else:
                cv2.circle(canvas, (x2, y2), 100, (255,255,255), -1)
                
            
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2

        # Now if the area is greater than the wiper threshold then set the clear variable to True and warn User.
        if area > wiper_thresh:
           cv2.putText(canvas,'Clearing Canvas',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5, cv2.LINE_AA)
           clear = True 

    
    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0

    cv2.imshow('image2',frame)

    frame = cv2.add(frame,canvas)

    # Optionally stack both frames and show it.
    # stacked = np.hstack((canvas,frame))
    # cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))
    # cv2.imshow('frame',frame)

    # Now this piece of code is just for smooth drawing. (Optional)
    _ ,mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)


    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 75, 0: 75] = eraser_img
    else:
        frame[0: 75, 0: 75] = pen_img

    # Merge the canvas and the frame.
    cv2.imshow('image',frame)
    # cv2.imshow('mask',mask)
    
    # cv2.imshow('foreground',foreground)
    # cv2.imshow('background',background)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None
    
    # Clear the canvas after 1 second if the clear variable is true
    if clear == True:
        
        time.sleep(1)
        canvas = None
        
        # And then set clear to false
        clear = False

cv2.destroyAllWindows()
cap.release()