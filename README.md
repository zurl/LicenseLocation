# 概述
这个项目实现了“车牌检测与识别”中的第一个子任务，使用的开发环境是

Windows10 + Anaconda 4.1.1 (with Python 2.7 Numpy Pandas Matplotlib) + OpenCV 3

项目提供了源代码，在detect.py中实现了detect_licese函数用于检测车牌，在detect_test.py中实现了对检测结果的正确性检验.

# 算法
该项目实现了一个基于机器视觉的途径的车牌检测算法,算法的流程如下:
1. 使用opencv的库函数将图片读入

2. 将图片resize为统一尺寸

3. 将图片转为灰度图

4. 对图片实施亮度增强算法

5. 对图片实施Y方向的Sobel边缘检测算法

6. 将图片二值化处理

7. [对图片实施噪音去除算法]

8. 对图片实施膨胀和侵蚀操作，并且迭代多次

9. 对图片进行轮廓检测操作

10. 对每个轮廓取外接旋转矩形

11. 对矩形进程过滤操作

12. 对每个旋转矩形取外界水平矩形，并且反映射为原图坐标并返回
该项目同时也实现了对检测算法的正确率评估算法，该算法的流程如下
1. 调用车牌检测算法，取得每组的坐标

2. 对结果进行目标包含测试

3. 对结果进行面积对比测试

4. 分别对第一顺位的结果和前三顺位的结果计算正确率
