import json
import numpy as np
from pprint import pprint
import cv2
import glob

for file in glob.glob('./image/' + "*.jpg"):
    file = file.split('/')[2].split('.')[0]

    with open('./pose/'+ file + '_keypoints.json') as f:
        data = json.load(f)
        joints = data["people"][0]["pose_keypoints"]
        joints = np.reshape(joints,(-1,3))
    print('./image/' + file + '.jpg')
    img = cv2.imread('./image/' + file + '.jpg')


    for i in range(len(joints)):
        print(joints[i])
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(joints[i][0]), int(joints[i][1]))
        fontScale = 0.3
        fontColor = (255, 255, 255)
        lineType = 1
        cv2.putText(img, str(i),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    cv2.imshow(file+'.jpg',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()