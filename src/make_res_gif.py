#===================================================
## Author: Reza Azad (rezazad68@gmail.com)
#===================================================

import numpy as np
import cv2
import glob, os

# Parameters
height = 726
width  = 1280

add = 'visualize/'

IMG_list = glob.glob(add+'*res2.png')
# IMG_list = os.listdir(add)
IMG_list = sorted(IMG_list)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

Output_video = cv2.VideoWriter('output.avi', fourcc, 5, (width, height ))

idx = 0
for fp in IMG_list:
    img = cv2.imread(fp)
    img = cv2.resize(img, (width, height))
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    Output_video.write(img)
    print(idx)
    idx += 1

Output_video.release()




