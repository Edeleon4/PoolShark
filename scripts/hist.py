import cv2
import numpy as np

frame = cv2.imread('/mnt/c/Users/T-HUNTEL/Desktop/hackathon/table3.jpg')
h,w,c = frame.shape
print frame.shape


# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

BORDER_COLOR = 0
def flood_fill(image, x, y, value):
    count = 1
    points = [(x, y)]
    "Flood fill on a region of non-BORDER_COLOR pixels."
    if x >= image.shape[1] or y >= image.shape[0] or image[x,y] == BORDER_COLOR:
        return None, None
    edge = [(x, y)]
    image[x, y] = value

    while edge:
        newedge = []
        for (x, y) in edge:
            for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if s <= image.shape[1] and y <= image.shape[0] and \
                	image[s, t] not in (BORDER_COLOR, value):
                    image[s, t] = value
                    points.append((s, t))
                    count += 1
                    newedge.append((s, t))
        edge = newedge

    return count, points


# thresholds for different balls / background
low_bkg = np.array([15, 40, 50], dtype=np.uint8)
high_bkg = np.array([40, 190, 200], dtype=np.uint8)

lower_blue = np.array([110,50,50], dtype=np.uint8)
upper_blue = np.array([130,255,255], dtype=np.uint8)

low_yellow = np.array([20, 30, 30], dtype=np.uint8)
high_yellow = np.array([30, 255, 255], dtype=np.uint8)


# mask out the background
mask = cv2.inRange(hsv, low_bkg, high_bkg)
mask = np.invert(mask)


# Bitwise-AND mask and original image
objects = cv2.bitwise_and(frame,frame, mask= mask)

hsv = cv2.cvtColor(objects, cv2.COLOR_BGR2HSV)

# mask the yellow balls
mask = cv2.inRange(hsv, low_yellow, high_yellow)

yellows = cv2.bitwise_and(objects, objects, mask=mask)

# find the biggest cloud of 1's in the yellow mask
biggest_cloud = []
biggest_count = 0

image = mask / 255.

while len(np.where(image == 1)[0]) > 0:
    loc = np.where(image == 1)
    y = loc[0][0]
    x = loc[1][0]
    count, cloud = flood_fill(image, y, x, 2)
    if count > biggest_count:
        print count
        biggest_count = count
        biggest_cloud = cloud

print biggest_cloud
print biggest_count

cv2.imwrite('mask.jpg', mask)
cv2.imwrite('yellows.jpg', yellows)
cv2.imwrite('frame.jpg', frame)
