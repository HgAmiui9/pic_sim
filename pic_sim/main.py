import os
import numpy as np
import imageio
import glob
import cv2
from PIL import Image

def read_png_file(imageList):
    images = []
    binarys = []
    diffs1 = []
    diffs2 = []
    opens = []
    
    for im_path in imageList:
        im = cv2.imread('out/'+im_path, flags=cv2.IMREAD_GRAYSCALE)
        resized_im = cv2.resize(im, (50, 50))
        # cv2.imshow("resized",resized_im)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # 
        _ret, binary = cv2.threshold(resized_im, 125, 255, 1)
        
        # save image
        cv2.imwrite('binarys/'+im_path, binary)
        
        
        images.append(resized_im)
        binarys.append(binary)
        
        
        kernel = np.ones((2, 2), dtype=np.uint8)
        # 2.噪声去除
        open = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        opens.append(open)
        # # 3.确定背景区域
        # sure_bg = cv2.dilate(open, kernel, iterations=3)
        # # 4.寻找前景区域
        # dist_transform = cv2.distanceTransform(open, 1, 5)
        # ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        # # 5.找到未知区域
        # sure_fg = np.uint8(sure_fg)
        # unknow = cv2.subtract(sure_bg, sure_fg)

        # # 6.类别标记
        # ret, markers = cv2.connectedComponents(sure_fg)
        # # 为所有的标记加1，保证背景是0而不是1
        # markers = markers + 1
        # # 现在让所有的未知区域为0
        # markers[unknow == 255] = 0

        # # 7.分水岭算法
        # markers = cv2.watershed(resized_im, markers)
        # resized_im[markers == -1] = (0, 0, 255)
            # Canny边缘检测

         # save image
        cv2.imwrite('resize/'+im_path, open)
        # blurred = cv2.GaussianBlur(src, (3, 3), 0)
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
 
        # grad_x = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
        # grad_y = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
 
        # dst = cv2.Canny(grad_x, grad_y, 30, 150)
        # # dst = cv.Canny(gray, 50, 150)
        # self.decode_and_show_dst(dst)
        
        # do whatever with the image here
    # cv2.imshow("resized",binarys[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(len(images))
    
    for i in range(int(len(images)/2)):
        cnt_im1 = opens[i*2]
        cnt_im2 = opens[i*2+1]
        dist = cv2.norm(cnt_im1 - cnt_im2, cv2.NORM_L2)
        
        diffs1.append(dist)
        
    print(diffs1)
    # print(diffs2)
    
        

if __name__=="""__main__""":
    # read png file
    imageList = os.listdir("out")
    
    imageList.remove('.DS_Store')
    imageList.sort()
    print(imageList)
    read_png_file(imageList)