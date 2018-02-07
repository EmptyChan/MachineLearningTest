# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 1/25/2018 2:47 PM 
 Description: 
"""
from os import listdir

import numpy as np


def createDataSet():
    """
    创建数据集和标签
     调用方式
     import kNN
     group, labels = kNN.createDataSet()
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(input_matrix, data_set, labels, k):
    """
    input_matrix: 用于分类的输入向量
    data_set: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    """
    # 计算距离, 行数
    data_set_size = data_set.shape[0]
    diff_matrix = np.tile(input_matrix, (data_set_size, 1)) - data_set
    """
    欧式距离,(a1-a2)^2 + (b1-b2)^2 + (c1-c2)^2的开方
    """
    sq_diff_matrix = diff_matrix ** 2
    # 相加
    sq_distances = sq_diff_matrix.sum(axis=1)
    # 开方
    distances = sq_distances ** 0.5
    # print(distances)
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sorted_distances_idnex = distances.argsort()
    # print(sorted_distances_idnex)

    # 2. 选择距离最小的k个点
    class_count = {}
    for i in range(k):
        # 找到该样本的类型
        vote_label = labels[sorted_distances_idnex[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_class_count[0][0]  # label


def file2matrix(file):
    with open(file) as f:
        lines = f.readlines()
        line_count = len(lines)
        mat = np.zeros((line_count, 3))
        clazzLabelVector = []
        for i, line in enumerate(lines):
            each_line_column_property = line.strip().split('\t')
            mat[i, :] = each_line_column_property[0:3]
            clazzLabelVector.append(int(each_line_column_property[-1]))
    return mat, clazzLabelVector


def autoNorm(data_set: np.ndarray):
    # 0代表第一象限，二维数组则代表是行，即固定行，看各个列中的最值
    min_value = data_set.min(0)
    print(min_value)
    max_value = data_set.max(0)
    print(max_value)
    # 极差
    ranges= max_value - min_value
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]  # 行数
    print(m)
    # 生成与最小值之差组成的矩阵
    norm_data_set = data_set - np.tile(min_value, (m, 1))  # 最小值重复行数m次行数，1次列数
    # print(norm_data_set)
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_value


def datingClassTest():
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    matrix, labels = file2matrix("./dating.txt")
    norm_matrix, ranges, min_value = autoNorm(matrix)
    # m 表示数据的行数，即矩阵的第一维
    m = norm_matrix.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    num_test_vecs = int(m * hoRatio)
    error = 0
    for i in range(num_test_vecs):
        # 对数据测试
        classifier_result = classify0(norm_matrix[i, :], norm_matrix[num_test_vecs:m, :], labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, labels[i]))
        if classifier_result != labels[i]:
            error += 1.0
    print(error)


def test1():
    """
    第一个例子演示
    """
    group, labels = createDataSet()
    # print(str(group))
    # print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))


def clasdifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('./dating.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


def img2vector(filename):
    """
    将图像数据转换为向量
    :param filename: 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    :return: 一维矩阵
    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    # 1. 导入数据
    hwLabels = []
    trainingFileList = listdir('input/2.KNN/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off.txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('input/2.KNN/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('input/2.KNN/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('input/2.KNN/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    # test1()
    # datingClassTest()
    clasdifyPerson()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title('kNN')
    # plt.xlabel('Frequent Flyier Miles Earned Per Year')
    # plt.ylabel('Percentage of Time Spent Playing Video Games')
    # x = matrix[:, 0]
    # y = matrix[:, 1]
    # s = 15.0 * np.array(labels)
    # ax.scatter(x, y, s=s, c=s)
    # plt.show()