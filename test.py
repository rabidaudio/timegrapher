import numpy as np
import cv2 as cv
from vidstab import VidStab

stabilizer = VidStab()

# https://drive.google.com/file/d/1Eu3DR6T4jKQukTe8VI3E8E9VK9Hy_ZB0/view?usp=share_link
# https://drive.google.com/file/d/1qBBdENsupzIjdF4CRR9DLdlt8VL0cuvp/view?usp=share_link

# cap = cv.VideoCapture('VID_20220802_202301.mp4')
cap = cv.VideoCapture('stab.avi')


def crop_square(img):
    [x, y, z] = img.shape
    size = min(x, y)
    x_offset = int((x - size) / 2)
    y_offset = int((y - size) / 2)
    return img[x_offset:x_offset+size, y_offset:y_offset+size, :]

back_sub = cv.createBackgroundSubtractorKNN()

f = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    f += 1

    # crop image to square
    frame = crop_square(frame)
    # grey scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # stabilized_frame = stabilizer.stabilize_frame(input_frame=gray,
    #                                                smoothing_window=30)
    # if f < 30:
    #     continue # use the first 30 frames for stabilization 

    motion_mask = back_sub.apply(gray)

    # search for circles in the middle 33% of the image
    # dim = int(stabilized_frame.shape[0]/3)
    # blurred = cv.blur(stabilized_frame, (5, 5))
    # # blurred = blurred[dim:dim+dim, dim:dim+dim]
    # circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 100, 100,
    #                             minRadius=250, maxRadius=500)
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0,:]:
    #         print()
    #         # draw the outer circle
    #         cv.circle(stabilized_frame, (i[0]+dim, i[1]+dim), i[2], (0,255,0), 2)
    #         # draw the center of the circle
    #         cv.circle(stabilized_frame, (i[0]+dim, i[1]+dim), 2, (0, 0, 255), 3)

    cv.imshow('frame', motion_mask)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

