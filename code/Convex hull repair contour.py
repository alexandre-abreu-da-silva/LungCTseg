import numpy as np
import cv2
import os
from skimage import io
np.set_printoptions(suppress=True)
#import tensorflow


def FillHole(mask):  #########白色为准
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def remove_longest_segments(contours):
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= 5000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:
                    if (length > 100 or length > ccccccccccccccc[1]):           #####去掉最长的1个  去掉
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)
        if area <=2000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:
                    if length > ccccccccccccccc[2] :  #####去掉最长的两个  去掉
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)


        if area > 2000 and area <5000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:

                    if (length > 70 or length > ccccccccccccccc[2]):  #####去掉最长的两个  去掉
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)







path_read = 'C:/Users/alex1/iCloudDrive/Mestrado/LungCTseg/data/processed_gradient_images'
path_gray = 'C:/Users/alex1/iCloudDrive/Mestrado/LungCTseg/data/contour_repaired_images'


for filename in os.listdir(path_read):
    print('正在进行的图片名称：', filename)
    img = io.imread(path_read + '/' + filename, 0)

    if np.max(img)==0:
        image_filtered=img
    else:
        img1 = img.copy()
        # 计算非0连通域数量
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img1, connectivity=8)


        areas = stats[1:, cv2.CC_STAT_AREA]  # 忽略背景连通域
        max_area_index = np.argmax(areas) + 1  # 获取除了背景以外的最大连通域的索引
        max_area = areas[max_area_index - 1]  # 获取最大连通域的面积
        print(max_area)
        if nLabels <= 3 and max_area > 2000:  ####     图较大，且存在需要连接起来的
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        if nLabels > 3 and max_area > 2000:  ####       图较大，需要连接起来的较多
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        if nLabels > 3 and max_area <= 2000:  ####      图较小
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if nLabels <= 3 and max_area <= 2000:  ####     图较小，且效果较好
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel1)
        # 查找轮廓并计算凸包
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        remove_longest_segments(contours)
        img1 = FillHole(img1)
        image_filtered=img1

        if type(img1)==int:
            image_filtered=np.zeros_like(img)
        else:
            pass







    cv2.imwrite(path_gray + '/' + filename,image_filtered)
