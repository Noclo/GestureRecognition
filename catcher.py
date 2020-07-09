# -*- coding: utf-8 -*- #

import os
import numpy as np
import cv2
import time
import cnn_train


# 提示语的位置、大小等参数
font = cv2.FONT_HERSHEY_SIMPLEX #　
size = 0.5 # 字体大小
fx = 10
fy = 380
fh = 18

# ROI 位置
x0 = 400
y0 = 100

# 输入网络的图片大小
width = 200
height = 200

# 录制手势的默认参数
# 每次录制多少张样本
num_of_samples = 400
# 计数器
counter = 0
# 存储地址和初始文件名称
gesture_name = ""
path = ""

#
guess_gesture = False   # 是否要决策的标志
last_gesture = -1 # 最后一帧图像

# 标识符 bool类型用来表示某些需要不断变化的状态
binaryMode = False # 是否将ROI显示为二值模式
saveImg = False # 是否保存图片


# banner = """\n choose a number: \n
# 1 - Training a net work and store the net.
# 2 - Record new gesture.
# """

def save_roi(img):
    # 保存ROI图像
    global path, counter, gesture_name, saveImg
    if counter > num_of_samples:
        # 恢复到初始值，以便后面继续录制手势
        saveImg = False
        gesture_name = ''
        counter = 0
        return

    counter += 1
    name = gesture_name + str(counter)   #给录制的手势命名
    print("Saving img: ", name)
    cv2.imwrite(path+name+'.png', img)   #写入文件
    time.sleep(0.01)

def binary_mask(frame, x0, y0, width, height):
    # 显示方框
    global guess_gesture, last_gesture, saveImg
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0,215,255),2)
    #提取ROI像素
    roi = frame[y0:y0+height, x0:x0+width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # 高斯模糊 斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
    # 高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大
    # 高斯模糊，给出高斯模糊矩阵和标准差

    # 显示ROI为二值模式
    # 图像的二值化，就是将图像上的像素点的灰度值设置为0或255，
    # 也就是将整个图像呈现出明显的只有黑和白的视觉效果。

    #  cv2.threshold  进行阈值化
    # 第一个参数  src     指原图像，原图像应该是灰度图
    # 第二个参数  x     指用来对像素值进行分类的阈值。
    # 第三个参数    y  指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # 有两个返回值 第一个返回值（得到图像的阈值）   二个返回值 也就是阈值处理后的图像
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 当同一幅图像上的不同部分的具有不同亮度时。    这种情况下我们需要采用自适应阈值
    # 参数： src 指原图像，原图像应该是灰度图。    x ：指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    # adaptive_method  指 CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C

    # 保存手势
    if saveImg == True and binaryMode == True:
        save_roi(res)
    elif saveImg == True and binaryMode == False:
        save_roi(roi)
    return res

def main():
    global x0, y0, binaryMode, saveImg, gesture_name, banner, guess_gesture, path

    cap = cv2.VideoCapture(0)# 创建一个视频捕捉对象
    while True:
        # 一帧一帧的捕捉视频
        ret, frame = cap.read()

        # 图像翻转  如果不翻转，视频中看起来的刚好和我们是左右对称的
        frame = cv2.flip(frame, 2)

        # 显示roi区域 #调用函数
        roi = binary_mask(frame, x0, y0, width, height)

        # 显示提示语
        cv2.putText(frame, "Option: ", (fx, fy), font, size, (0,215,255),2)
        cv2.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (fx, fy + fh), font, size, (0,215,255),2)
        cv2.putText(frame, "s-'new gestures(twice)'", (fx, fy + 2 * fh), font, size, (0,215,255),2)
        cv2.putText(frame, "q-'quit'", (fx, fy + 3 * fh), font, size, (0,215,255),2)


        key = cv2.waitKey(1) & 0xFF #等待键盘输入
        if key == ord('b'):
            # binaryMode = not binaryMode
            binaryMode = True
            print("Binary Threshold filter active")
        elif key == ord('r'):
            binaryMode = False

        # 调整ROI框
        if key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

        if key == ord('q'):
            break
        if key == ord('s'):
            """录制新的手势（训练集）"""
            if gesture_name != "":  #
                saveImg = True
            else:
                print("Enter a gesture group name first, by enter press 'n'! ")
                saveImg = False
        elif key == ord('n'):
            # 开始录制新手势
            # 首先输入文件夹名字
            gesture_name = input("enter the gesture folder name: ")
            os.makedirs(gesture_name)
            path = "./" + gesture_name + "/" # 生成文件夹的地址  用来存放录制的手势


        # 展示处理之后的视频帧
        cv2.imshow("frame", frame)

        if binaryMode:
            cv2.imshow("ROI", roi)
        else:
            cv2.imshow("ROI", frame[y0:y0+height, x0:x0+width])


    #最后记得释放捕捉
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





