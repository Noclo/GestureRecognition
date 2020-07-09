## 利用卷积神经网络实时识别手势动作

一共识别7种手势动作
1. ok 2. peace 3. good 4. punch 5. love 6. heart 7.stop

### 项目文件

项目文件列表如下：

1. `img_b`：放置处理完的图像
2. `log`：存放训练的CNN网络的模型参数
3. `model`：存放训练好的模型
4. `cnn_train.py`:初始化数据，训练数据测试的分类，搭建网络参数，定义前向卷积，训练模块，，以及预测模块程序
5. `cnn_text.py`:测试模块程序
6. `catcher.py`：显示roi图像，摄像框的调出及图像数据处理，录制新的手势
7. `prediction.py`： 实时预测


### 使用方法

运行catcher.py调出摄像头进行录制手势
运行cnn_train.py 训练数据
运行cnn_test.py 测试模型
运行prediction.py进行预测


### 测试结果：
使用该模型训练到2400步的时候在测试集上正确率可以稳定在97%左右。







