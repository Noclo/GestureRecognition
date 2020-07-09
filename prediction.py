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

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


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

        # if key == ord('p'):
        #     """调用模型开始预测, 对二值图像预测，所以要在二值函数里面调用，预测新采集的手势"""
        #
        #     roi = np.reshape(roi, [width, height, 1])
        #     gesture = myCNN.guess(roi)
        #     gesture_copy = gesture
        #     cv2.putText(frame, gesture_copy, (10, 150), font, 2, (0, 0, 255), 6, 8)  # 标注字体de
            # while True:
            #    myCNN.guess(roi)


        if key == ord('q'):
            break

        roi = np.reshape(roi, [width, height, 1])
        gesture = cnn_train.guess(roi)
        gesture1_copy = gesture
        cv2.putText(frame, gesture1_copy, (50, 200), font, 2, (0,215,255), 6, 8)  # 标注字体
        time.sleep(0.01)

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
