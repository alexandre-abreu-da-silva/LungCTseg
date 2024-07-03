import numpy as np
import cv2
# import SegmentationMetric
import os
from skimage import io
np.set_printoptions(suppress=True)
import tensorflow


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
        # print(area)
        if area >= 5000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            # print(ccccccccccccccc)
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
            # print(ccccccccccccccc)
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
            # print(ccccccccccccccc)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:
                    #                10000                         100  1      70会↑                                      50会↓
                    # Avgccuracy = 0.9910543073307382                      0.9910481626337223                         0.9910294774806858
                    # AvgPrecision = 0.9691889721037688                    0.9697217372936162                         0.9703261996505118
                    # AvgRecall = 0.9525615317128047                       0.9522056109978514                         0.9516052055763777
                    # AvgMIoU = 0.9591807907161727                         0.9591949026614858                         0.9591496184062065                                               0.9516052055763777
                    # AvgF1Score = 0.9595171071231848                      0.9595466687349687                         0.959495947757146
                    #            10000                                      0.991031431790554      70 2   (80 2)↓
                    #                                                       0.9707169692046083
                    #                                                       0.951481454894498
                    #                                                       0.9592602008768885
                    #                                                       0.9596313853455704
                    #            5000                 2000 2                        6000                 70 2
                    #       0.9910812721107943   0.9910874746062537        0.9910830494129295
                    #       0.9700465242872675   0.9705234473685888         0.9701683289447208
                    #       0.9524767898360711   0.9523801725436207         0.9523707130444693
                    #       0.9594729594367681    0.9595538950151162        0.9594774937684755
                    #       0.9598592766058884    0.9599842009203763        0.9598554427196857
                    if (length > 70 or length > ccccccccccccccc[2]):  #####去掉最长的两个  去掉
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)



#15 30      100
# 平均准确率为： 0.9910655870582116
# 平均精度为： 0.9691411992850785
# 平均召回率为： 0.9534142273113491
# 平均 MIoU 为： 0.9594844676462497
# 平均 F1 Score 为： 0.9600031001159707
#323
# 平均准确率为： 0.9910519845557934
# 平均精度为： 0.9689959885607801
# 平均召回率为： 0.9525960440355202
# 平均 MIoU 为： 0.959126893354949
# 平均 F1 Score 为： 0.9594153412694832



path_read = 'E:/pythonProject/kaggle/canny/prediction_bi/'
path_gray = 'E:/pythonProject/kaggle/canny/prediction_convex/'


for filename in os.listdir(path_read):
    print('正在进行的图片名称：', filename)
    img = io.imread(path_read + filename, 0)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

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
        # if nLabels <= 2 and max_area >=2000:  ####     图的效果很好
        #     kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
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







    cv2.imwrite(path_gray + filename,image_filtered)
