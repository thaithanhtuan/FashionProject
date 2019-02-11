import cv2
import ntpath
import json
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
#Visualize for training data, model and cloth on same image name
file = open("/media/emcom/RESEARCH/Dataset/listfilepose1.txt", "r")
fig=plt.figure(figsize=(192, 256))
columns = 3
rows = 2
for line in tqdm(file):
    # copy original images
    #print(line)
    image = cv2.imread("data/train/image/" + \
          ntpath.basename(line).rstrip().split(".")[0] + ".jpg")
    warp_top_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/top/" + \
          ntpath.basename(line).rstrip().split(".")[0] + ".jpg")
    warp_bottom_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/bottom/" + \
                     ntpath.basename(line).rstrip().split(".")[0] + ".jpg")
    warp_top_mask = cv2.imread("result/step_040000.pth/test/warp-mask/top/" + \
                     ntpath.basename(line).rstrip().split(".")[0] + ".png")
    warp_bottom_mask = cv2.imread("result/step_040000.pth/test/warp-mask/bottom/" + \
                    ntpath.basename(line).rstrip().split(".")[0] + ".png")
    affine_top_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/top/" + \
                     ntpath.basename(line).rstrip().split(".")[0] + ".jpg")
    affine_bottom_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/bottom/" + \
                                  ntpath.basename(line).rstrip().split(".")[0] + ".jpg")
    image_top_bottom = (0.4 * warp_top_cloth + 0.2 * image + 0.4 * warp_bottom_cloth).astype(np.uint8)
    image_top = (0.8 * warp_top_cloth + 0.2 * image).astype(np.uint8)
    image_bottom = (0.8 * warp_bottom_cloth + 0.2 * image).astype(np.uint8)

    line0 = np.hstack((image, affine_top_cloth, affine_bottom_cloth))
    line1 = np.hstack((image,warp_top_cloth,warp_bottom_cloth))
    line2 = np.hstack((image_top_bottom, image_top, image_bottom))
    final = np.vstack((line0,line1,line2))
    #cv2.imshow('Visualize top and bottom warped cloth', final)
    cv2.imwrite("result/step_040000.pth/test/visualize/" + \
                    ntpath.basename(line).rstrip().split(".")[0] + ".jpg",final)
    #exit()

"""

"""
#Visualize for testing data, model on image ith, cloth on image (i+1)th
file = open("/media/emcom/RESEARCH/Dataset/listfilepose1.txt", "r")
fig=plt.figure(figsize=(192, 256))
columns = 3
rows = 2
list_file = []
for line in tqdm(file):
    list_file.append(line)
for i in tqdm(range(len(list_file)-1)):
    image = cv2.imread("data/train/image/" + \
                       ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg")
    warp_top_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/top/" + \
                                ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    warp_bottom_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/bottom/" + \
                                   ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    warp_top_mask = cv2.imread("result/step_040000.pth/test/warp-mask/top/" + \
                               ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".png")
    warp_bottom_mask = cv2.imread("result/step_040000.pth/test/warp-mask/bottom/" + \
                                  ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".png")
    affine_top_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/top/" + \
                                  ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    affine_bottom_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/bottom/" + \
                                     ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    image_top_bottom = (0.4 * warp_top_cloth + 0.2 * image + 0.4 * warp_bottom_cloth).astype(np.uint8)
    image_top = (0.8 * warp_top_cloth + 0.2 * image).astype(np.uint8)
    image_bottom = (0.8 * warp_bottom_cloth + 0.2 * image).astype(np.uint8)

    line0 = np.hstack((image, affine_top_cloth, affine_bottom_cloth))
    line1 = np.hstack((image, warp_top_cloth, warp_bottom_cloth))
    line2 = np.hstack((image_top_bottom, image_top, image_bottom))
    final = np.vstack((line0, line1, line2))
    # cv2.imshow('Visualize top and bottom warped cloth', final)
    cv2.imwrite("result/step_040000.pth/test/visualize/" + \
                ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg", final)
image = cv2.imread("data/train/image/" + \
          ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg")
warp_top_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/top/" + \
      ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
warp_bottom_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/bottom/" + \
                 ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
warp_top_mask = cv2.imread("result/step_040000.pth/test/warp-mask/top/" + \
                 ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".png")
warp_bottom_mask = cv2.imread("result/step_040000.pth/test/warp-mask/bottom/" + \
                ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".png")
affine_top_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/top/" + \
                 ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
affine_bottom_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/bottom/" + \
                              ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
image_top_bottom = (0.4 * warp_top_cloth + 0.2 * image + 0.4 * warp_bottom_cloth).astype(np.uint8)
image_top = (0.8 * warp_top_cloth + 0.2 * image).astype(np.uint8)
image_bottom = (0.8 * warp_bottom_cloth + 0.2 * image).astype(np.uint8)

line0 = np.hstack((image, affine_top_cloth, affine_bottom_cloth))
line1 = np.hstack((image,warp_top_cloth,warp_bottom_cloth))
line2 = np.hstack((image_top_bottom, image_top, image_bottom))
final = np.vstack((line0,line1,line2))
#cv2.imshow('Visualize top and bottom warped cloth', final)
cv2.imwrite("result/step_040000.pth/test/visualize/" + \
                ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg",final)
"""

"""
#Visualize no background image
"""
file = open("/media/emcom/RESEARCH/Dataset/listfilepose1.txt", "r")
fig=plt.figure(figsize=(192, 256))
columns = 3
rows = 2
list_file = []
for line in tqdm(file):
    list_file.append(line)
for i in tqdm(range(len(list_file)-1)):
    image = cv2.imread("data/train/imagenobg/" + \
                       ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg")
    image_cloth = cv2.imread("data/train/imagenobg/" + \
                       ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")

    warp_top_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/top/" + \
                                ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    warp_bottom_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/bottom/" + \
                                   ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    warp_timage_clothop_mask = cv2.imread("result/step_040000.pth/test/warp-mask/top/" + \
                               ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".png")
    warp_bottom_mask = cv2.imread("result/step_040000.pth/test/warp-mask/bottom/" + \
                                  ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".png")
    affine_top_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/top/" + \
                                  ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    affine_bottom_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/bottom/" + \
                                     ntpath.basename(list_file[i+1]).rstrip().split(".")[0] + ".jpg")
    alpha_top_cloth = cv2.imread("result/step_180000.pth/test/alpha_top_dir/" + \
                                     ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg")
    alpha_bottom_cloth = cv2.imread("result/step_180000.pth/test/alpha_bottom_dir/" + \
                                     ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg")
    try_on_cloth = cv2.imread("result/step_180000.pth/test/try-on/" + \
                                     ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg")
    render_cloth = cv2.imread("result/step_180000.pth/test/render_dir/" + \
                                     ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg")
    image_top_bottom = (0.4 * warp_top_cloth + 0.2 * image + 0.4 * warp_bottom_cloth).astype(np.uint8)
    image_top = (0.8 * warp_top_cloth + 0.2 * image).astype(np.uint8)
    image_bottom = (0.8 * warp_bottom_cloth + 0.2 * image).astype(np.uint8)

    #print(image.shape, ":", try_on_cloth.shape)
    line0 = np.hstack((image_cloth,affine_top_cloth, warp_top_cloth, alpha_top_cloth))
    line1 = np.hstack((image_cloth,affine_bottom_cloth, warp_bottom_cloth, alpha_bottom_cloth))
    line2 = np.hstack((image_cloth,image_top_bottom, image_top, image_bottom))
    line3 = np.hstack((image_cloth,image, render_cloth, try_on_cloth))

    final = np.vstack((line0, line1, line2, line3))
    # cv2.imshow('Visualize top and bottom warped cloth', final)
    cv2.imwrite("result/step_180000.pth/test/visualize/" + \
                ntpath.basename(list_file[i]).rstrip().split(".")[0] + ".jpg", final)
image = cv2.imread("data/train/imagenobg/" + \
                       ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg")
warp_top_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/top/" + \
                            ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
warp_bottom_cloth = cv2.imread("result/step_040000.pth/test/warp-cloth/bottom/" + \
                               ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
warp_top_mask = cv2.imread("result/step_040000.pth/test/warp-mask/top/" + \
                           ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".png")
warp_bottom_mask = cv2.imread("result/step_040000.pth/test/warp-mask/bottom/" + \
                              ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".png")
affine_top_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/top/" + \
                              ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
affine_bottom_cloth = cv2.imread("result/step_040000.pth/test/affine-warp-cloth/bottom/" + \
                                 ntpath.basename(list_file[0]).rstrip().split(".")[0] + ".jpg")
alpha_top_cloth = cv2.imread("result/step_180000.pth/test/alpha_top_dir/" + \
                                 ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg")
alpha_bottom_cloth = cv2.imread("result/step_180000.pth/test/alpha_bottom_dir/" + \
                                 ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg")
try_on_cloth = cv2.imread("result/step_180000.pth/test/try-on/" + \
                                 ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg")
render_cloth = cv2.imread("result/step_180000.pth/test/render_dir/" + \
                                 ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg")
image_top_bottom = (0.4 * warp_top_cloth + 0.2 * image + 0.4 * warp_bottom_cloth).astype(np.uint8)
image_top = (0.8 * warp_top_cloth + 0.2 * image).astype(np.uint8)
image_bottom = (0.8 * warp_bottom_cloth + 0.2 * image).astype(np.uint8)

line0 = np.hstack((affine_top_cloth, warp_top_cloth, alpha_top_cloth))
line1 = np.hstack((affine_bottom_cloth, warp_bottom_cloth, alpha_bottom_cloth))
line2 = np.hstack((image_top_bottom, image_top, image_bottom))
line3 = np.hstack((image, render_cloth, try_on_cloth))

final = np.vstack((line0, line1, line2, line3))
# cv2.imshow('Visualize top and bottom warped cloth', final)
cv2.imwrite("result/step_180000.pth/test/visualize/" + \
            ntpath.basename(list_file[len(list_file)-1]).rstrip().split(".")[0] + ".jpg", final)

