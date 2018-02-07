# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 1/31/2018 10:28 AM 
 Description: 
"""
from math import log
from decision_tree_plot import createPlot

print(__doc__)


def create_data_set():
    """DateSet 基础数据集
    Returns:
        返回数据集和对应的label标签
    """
    data_set = [['帅', '豪', '高', '嫁'],
                ['帅', '豪', '矮', '嫁'],
                ['帅', '穷', '高', '不嫁'],
                ['丑', '豪', '矮', '嫁'],
                ['丑', '穷', '矮', '不嫁']]
    labels = ['长相', '有钱', '身高']
    return data_set, labels


def calculate_shannon_entropy(data_set):
    """calcShannonEnt(calculate Shannon entropy 计算给定数据集的香农熵)
    Args:
        dataSet 数据集
    Returns:
        返回 每一组feature下的某个分类下，香农熵的信息期望
    """
    # -----------计算香农熵的第一种实现方式start--------------------------------------------------------------------------------
    # 求数据集的行数，即当前固定的特征下的数据行数
    num_entries = len(data_set)
    # 计算分类标签label出现的次数
    label_count = {}
    for feature_vector in data_set:
        # 将当前的标签存储，每一行数据的最后一个数据是标签
        current_label = feature_vector[-1]
        label_count.setdefault(current_label, 0)
        label_count[current_label] += 1
    # 对于label标签的占比，求出label标签的信息熵
    shannon_entropy = 0.0
    for k, v in label_count.items():
        # 使用固定特征下或者综合情况下， 类标签的发生频率计算类别出现的概率
        prob = float(v) / num_entries
        # 计算信息熵，以2位底数
        shannon_entropy -= prob * log(prob, 2)
    print('信息熵' + str(shannon_entropy))
    # -----------计算香农熵的第一种实现方式end--------------------------------------------------------------------------------

    # # -----------计算香农熵的第二种实现方式start--------------------------------------------------------------------------------
    # # 统计各个标签出现的次数
    # label_count = Counter(data[-1] for data in data_set)
    # probs = [v / num_entries for k, v in label_count.items()]
    # shannon_entropy = sum([-p * log(p, 2) for p in probs])
    # print(shannon_entropy)
    return shannon_entropy
    # # -----------计算香农熵的第二种实现方式end--------------------------------------------------------------------------------


def split_data_set(data_set, index, value):
    """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
        其实就是按照固定的特征切分数据集，返回包含该特征的数据集，同时去除该特征列
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    # 第一种方式
    return_data_set = []
    for feature_vector in data_set:
        # 判断index特征列是不是等于value
        if feature_vector[index] == value:
            reduce_feature_vector = feature_vector[:index] + feature_vector[index + 1:]
            return_data_set.append(reduce_feature_vector)

    # 第二种方式
    # return_data_set = [data for data in data_set for i, v in enumerate(data) if i == index and v == value]
    return return_data_set


def choose_best_feature(data_set):
    """chooseBestFeatureToSplit(选择最好的特征)
    Args:
       dataSet 数据集
    Returns:
       bestFeature 最优的特征列
    """
    # -----------选择最优特征的第一种方式 start------------------------------------
    # 求数据集合有多少列数的特征, 最后一列是label
    num_feature = len(data_set[0]) - 1
    # 原始的信息熵
    base_entropy = calculate_shannon_entropy(data_set)
    # 最佳信息增益和最优的特征index
    base_info_gain, best_feature = 0.0, -1
    for i in range(num_feature):
        # 获取下标是i的特征列所有的特征表示情况, 例如长相，有帅和丑2种情况
        current_datas = [example[i] for example in data_set]
        # 剔除重复值，保留所有属性状态
        unique_values = set(current_datas)
        # 创建一个临时熵，针对当前特征
        conditional_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)  # 固定下标为i的特征后，得到单独的属性的子集（例如长相为帅的子集）
            prob = len(sub_data_set) / float(len(data_set))  # 获取长相为帅的属性值占所有行数的概率
            temp_entropy = calculate_shannon_entropy(sub_data_set)  # 即固定长相为其中某一个属性，比如帅时，它的信息熵是多少
            # 计算条件熵
            conditional_entropy += prob * temp_entropy
        # 信息增益等于原始信息熵减去固定某一个特征后的条件熵，我们的目的是获取最大的信息增益
        info_gain = base_entropy - conditional_entropy
        print(str(i) + '>>>>>>' + str(info_gain))
        if info_gain > base_info_gain:
            base_info_gain = info_gain  # 赋值最佳信息增益
            best_feature = i  # 获取最佳特征
    # print(base_info_gain)
    # print(best_feature)
    return best_feature
    # # -----------选择最优特征的第二种方式 start------------------------------------
    # # 计算初始香农熵
    # base_entropy = calcShannonEnt(dataSet)
    # best_info_gain = 0
    # best_feature = -1
    # # 遍历每一个特征
    # for i in range(len(dataSet[0]) - 1):
    #     # 对当前特征进行统计
    #     feature_count = Counter([data[i] for data in dataSet])
    #     # 计算分割后的香农熵
    #     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
    #                    for feature in feature_count.items())
    #     # 更新值
    #     info_gain = base_entropy - new_entropy
    #     print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
    #     if info_gain > best_info_gain:
    #         best_info_gain = info_gain
    #         best_feature = i
    # return best_feature
    # # -----------选择最优特征的第二种方式 end------------------------------------


def majority_cnt(class_list):
    """majorityCnt(选择出现次数最多的一个结果)
    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """
    # -----------majorityCnt的第一种方式 start------------------------------------
    classCount = {}
    for vote in class_list:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    # print 'sortedClassCount:', sortedClassCount
    return sortedClassCount[0][0]
    # -----------majorityCnt的第一种方式 end------------------------------------

    # # -----------majorityCnt的第二种方式 start------------------------------------
    # major_label = Counter(classList).most_common(1)[0]
    # return major_label
    # # -----------majorityCnt的第二种方式 end------------------------------------


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优的列，得到最优列对应的label含义
    best_feature_index = choose_best_feature(data_set)
    # 获取label的名称
    best_label = labels[best_feature_index]
    # 初始化tree
    my_tree = {best_label: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[best_feature_index])
    # 取出最优列，然后它的branch做分类
    feature_values = [example[best_feature_index] for example in data_set]  # 假设最优列是长相
    unique_list = set(feature_values)
    for unique in unique_list:
        # 求出剩余的label
        sub_labels =labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()， 分别创建子树
        # 即分割长相这个最优列后寻找新的最优列
        my_tree[best_label][unique] = create_tree(split_data_set(data_set, best_feature_index, unique), sub_labels)
    return my_tree


def classify(inputTree, featLabels, testVec):
    """classify(给输入的节点，进行分类)
    Args:
        inputTree  决策树模型
        featLabels Feature标签对应的名称
        testVec    测试输入的数据
    Returns:
        classLabel 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++' + str(firstStr) + 'xxx' + str(secondDict) + '---' + str(key) + '>>>' + str(valueOfFeat))
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = create_data_set()
    # print myDat, labels

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = create_tree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, ['丑', '豪', '高']))

    # 画图可视化展现
    createPlot(myTree)


if __name__ == '__main__':
    fishTest()