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

# 效果检测

	这里并没有采用PPT中提供的方法进行校检，本项目的效果检测分为两部分：
	1.包含测试：检测标记点是否全在算法得到的矩形中
	2.面积对比：对比标记点构成矩形面积与算法得到的矩阵的面积
	这里由于算法具有局限性，在面积系数使用了较大的6.8倍，同时计算corr1与corr3，分别是在算法给出结果集的第一个和前三种是否拥有通过测试的矩形的数目的比例。

# 核心算法

## 亮度增强算法

	根据论文[1]给出的算法，根据车牌的亮度特征（均值与标准差），利用二次线性插值方法简化计算，通过控制由标准差决定的亮度增强系数，达到将车牌区域亮度提高的效果。
	在样本数据集的一部分数据（[350:500]）上，亮度增强算法起到了显著的作用，Corr1由14%提高到41.3%,Corr3由14.7%提高到47.3%

## 噪音消除算法

	根据论文[1]给出的算法，通过使用CNP可以消除杂边，但是实际上该算法在大多数参数下并不奏效，故最终没有默认采纳该算法。

## Sobel边缘检测算法

	项目中根据论文[2]启发，使用了openCV提供的sobel边缘检测算法，sobel算法可以指定方向，由于车牌存在大量y方向的线条，故在y方向应用sobel算法，取得了良好效果。

## 形态学开闭运算

	通过边缘检测算法的图形存在大量零散线条，需要将其聚合，同时也存在大量无用信息，这时根据论文[2]的启发，使用openCV提供的膨胀和腐蚀操作，可以有效的将车牌的线条聚合为一个区域。

## 轮廓检测操作

	根据openCV文档的指示，使用openCV库函数对开闭运算后的图像运行轮廓检测，对轮廓寻找外接旋转矩形。


## 参考文献
	[1]An efficient method of license plate location，Danian Zheng, Yannan Zhao, Jiaxin Wang
	[2]Approach of Car License Plate Location based on Improved Sobel Operator with the Combination of Mathematical Morphology，YanlingCui ， Chengjun Yang

# 后记

	1. 为什么要选择这个项目
	群里一共提供了三个项目，一个歌曲推荐，一个车牌检测还有一个算法实现，我都一一的阅读了一下，
	因为我之前没有任何这方面的基础，也一点不了解图像处理有关的技术，所以在我读了一下推荐的入门读物之后，选择了一个貌似和机器学习很有关的歌曲推荐项目，并且简单的实现了一个协同过滤算法（这里不引用原文了），但是这个算法需要维护一个与歌曲数目N有关的N*N矩阵，但是歌曲数量着实巨大，我在选择了其中800首作为样本之后，很遗憾正确率是0，这令我十分灰心……后来觉得这可能是我的电脑的计算能力的问题，而后就准备尝试一下车牌检测的项目，其中很自然的就想到先尝试第一个任务。
	2. 为什么选择Python
	首先是我对Python比较了解，在这之前没有尝试过Matlab，python也是一个比较流行的语言，其次在这个项目中，使用了OpenCV这个库对C++与Python支持较好，而C++开发太慢所以就选择了Python2作为这次的开发语言。
	3. 为什么选择OpenCV
	之前听说过OpenCV是做机器视觉的，所以当准备做这个项目时，首先就先通读了一下openCV 3 offical document 的概述和目录，发现这个库对这次项目很可能有非常大帮助，于是就选择了它。
	4. 感觉中国人在这个研究方向非常感兴趣。
	5. 在百度上我也找到了几篇有关车牌识别的博客之类的文章，但是并没有采用，有一篇采用TensorFlow的github项目看起来很靠谱，但是电脑配置有限……估计不是很能实现，还有一篇采用OpenCV自带的分类器的方法，不过这篇文章内容比较概括，我也不是非常理解，很难实现，但是后来发现了[1]这篇论文，感觉前半部分讲的非常清楚，通过这篇文章也大概理解了整个过程是大概什么样的情况，但是这篇文章后半部分的内容不是很清晰，所以我只好尝试其他办法。
	6. 后来发现了[2]文章，这篇文章的形态学方法看起来和容易利用openCV实现，就把它实现了，后来发现其实效果比较一般（因为噪音很大），于是就按照[1]的方法做了杂边过滤，但是效果反倒更加糟糕，非常尴尬。
	7. 后来在《学习openCV》这本书上介绍的形态学内容调整了一下算法，感觉有明显改进，调整了一下参数，发现瓶颈在于有的车牌比较暗，就调整了一下亮度增强算法的参数，发现效果有明显提升……
	8. 最后的轮廓检测算法和外接矩形都是在OpenCV文档上找到的，感觉文档都是很靠谱的。
	9. 我感觉算法的效果很糟糕，也不是很清楚怎么提高它，目前只能调参……比较尴尬
	10. 最后发现这个项目貌似和机器学习关系不大………………但是学到了openCV应该怎么用感觉收获还是很大的...

