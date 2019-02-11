import glob
from PIL import Image
import json
import numpy as np



"""
listdir = ["data/train/cloth/bottom","data/train/cloth/top","data/train/cloth-mask/bottom",
           "data/train/cloth-mask/top","data/train/image/","data/train/image-parse",
           "data/train/warp-cloth/bottom","data/train/warp-cloth/top",
           "data/train/warp-mask/bottom","data/train/warp-mask/top"]
#resize image
for dir in listdir:
    files=glob.glob(dir+"/*.jpg")
    files.extend(glob.glob(dir+"/*.png"))
    for file in files:
        print(file)
        img = Image.open(file)
        img1 = img.resize((192,256), Image.NEAREST)
        img1.save(file)
     
"""
"""
#resize imagenobg and imagewithbg
"""
listdir = ["data/train/imagenobg","data/train/imagewithbg",
           "data/test/imagenobg","data/test/imagewithbg"]
for dir in listdir:
    files = glob.glob(dir + "/*.jpg")
    files.extend(glob.glob(dir + "/*.png"))
    for file in files:
        print(file)
        img = Image.open(file)
        img1 = img.resize((192, 256), Image.NEAREST)
        img1.save(file)

#resize pose
"""
dir = "data/train/pose/"
for file in glob.glob(dir + "*.json"):
    print(file)
    with open(file, 'r+') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))
        # print(":::", pose_data)
        pose_data = pose_data * [192.0 / 400.0, 256.0 / 600.0, 1]
        pose_data = pose_data.flatten()
        pose_label['people'][0]['pose_keypoints'] = pose_data.tolist()
        f.seek(0)
        json.dump(pose_label, f)
        f.truncate()
"""
#change background color of cloth
"""
listdir = ["data/train/cloth/bottom","data/train/cloth/top",
           "data/train/warp-cloth/bottom","data/train/warp-cloth/top"]


for dir in listdir:
    files=glob.glob(dir+"/*.jpg")
    files.extend(glob.glob(dir+"/*.png"))
    for file in files:
        print(file)
        img = Image.open(file)
        pixdata = img.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if pixdata[x, y] == (0,0,0):
                    pixdata[x, y] = (255,255,255)

        img.show()
        exit()



"""
