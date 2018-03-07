# -*- coding: utf-8 -*- 
"""
 Created with IntelliJ IDEA.
 Description:
 User: jinhuichen
 Date: 2/28/2018 11:12 AM 
 Description: 剪枝
              http://blog.csdn.net/yujianmin1990/article/details/49864813
              https://www.tuicool.com/articles/2Inaam
              http://www.cnblogs.com/starfire86/p/5749334.html
              http://blog.csdn.net/qq_20282263/article/details/52718532
              https://www.jianshu.com/p/794d08199e5e
              http://blog.csdn.net/o1101574955/article/details/50371499
"""
import copy
import io
from math import log, sqrt

import sys

from machine_learning.decision_tree_plot import createPlot
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def create_data_set():
    """DateSet 基础数据集
    Returns:
        返回数据集和对应的label标签
    """
    data_set = [['帅', '豪', '高', 24, '嫁'],
                ['帅', '豪', '矮', 26, '不嫁'],
                ['帅', '穷', '高', 50, '不嫁'],
                ['丑', '豪', '矮', 40, '嫁'],
                ['丑', '穷', '矮', 26, '不嫁'],
                ['帅', '穷', '矮', 27, '不嫁'],
                ['丑', '穷', '高', 30, '嫁'],
                ['帅', '穷', '高', 50, '不嫁']]
    labels = ['长相', '有钱', '身高', '年龄']
    # types = [0, 0, 0, 1]
    return data_set, labels  # , types


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
    # print('信息熵' + str(shannon_entropy))
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


def split_continuous_data_set(data_set, index, split_value):
    less_than_data_set = []
    greater_than_and_equal_data_set = []
    for feature_vector in data_set:
        if feature_vector[index] < split_value:
            less_than_data_set.append(feature_vector[:index] + feature_vector[index + 1:])
        elif feature_vector[index] >= split_value:
            greater_than_and_equal_data_set.append(feature_vector[:index] + feature_vector[index + 1:])
    return less_than_data_set, greater_than_and_equal_data_set


def choose_best_feature(data_set, labels):
    """chooseBestFeatureToSplit(选择最好的特征)
    Args:
       dataSet 数据集
       labels 标签
    Returns:
       bestFeature 最优的特征列
    """
    # -----------选择最优特征的第一种方式 start------------------------------------
    # 求数据集合有多少列数的特征, 最后一列是label
    num_feature = len(data_set[0]) - 1
    clazz_set = [ex[-1] for ex in data_set]
    # 原始的信息熵
    base_entropy = calculate_shannon_entropy(data_set)
    # 最佳信息增益和最优的特征index
    base_info_gain, best_feature = 0.0, -1
    best_split_dot = {}
    # 最佳信息增益率
    best_info_gain_ratio = 0
    for i in range(num_feature):
        # 获取下标是i的特征列所有的特征表示情况, 例如长相，有帅和丑2种情况
        current_datas = [example[i] for example in data_set]
        # 连续属性值的分裂
        if type(current_datas[0]).__name__ == 'float' or type(current_datas[0]).__name__ == 'int':
            sort_data = sorted(current_datas, key=lambda item: item)
            internal_best_info_gain = 0.0
            internal_own_entropy = 0.0
            internal_best_own_entropy = internal_own_entropy
            internal_conditional_entropy = 0.0
            split_index = 0  # 用来分裂连续型属性的分裂点数目，表示有几种分裂方式，用来修正决策树偏向于选择连续性属性
            best_split_middle_value = 0.0
            for j in range(len(sort_data) - 1):
                if clazz_set[j] != clazz_set[j + 1]:
                    split_index += 1
                    middle_value = float(sort_data[j] + sort_data[j + 1]) / 2
                    less_than_data_set, greater_than_and_equal_data_set = split_continuous_data_set(data_set, i,
                                                                                                    middle_value)
                    # 计算分裂信息度量
                    less_prob = len(less_than_data_set) / float(len(data_set))
                    greater_prob = len(greater_than_and_equal_data_set) / float(len(data_set))
                    internal_own_entropy -= less_prob * log(less_prob, 2)
                    internal_own_entropy -= greater_prob * log(greater_prob, 2)
                    if internal_own_entropy == 0:  # 分裂属性度量为0
                        continue
                    # 各自计算条件熵
                    internal_conditional_entropy += less_prob * calculate_shannon_entropy(less_than_data_set)
                    internal_conditional_entropy += greater_prob * calculate_shannon_entropy(
                        greater_than_and_equal_data_set)

                    # 获取内部最优的信息增益
                    internal_info_gain = base_entropy - internal_conditional_entropy
                    if internal_info_gain > internal_best_info_gain:
                        internal_best_info_gain = internal_info_gain
                        internal_best_own_entropy = internal_own_entropy
                        best_split_middle_value = middle_value
            best_split_dot.setdefault(labels[i], best_split_middle_value)
            fix_gain = log(split_index, 2) / len(data_set)
            info_gain_ratio = (internal_best_info_gain - fix_gain) / float(internal_best_own_entropy)
        else:
            # 剔除重复值，保留所有属性状态
            unique_values = set(current_datas)
            # 创建一个临时熵，针对当前特征
            conditional_entropy = 0.0
            # 分裂信息度量，即惩罚参数，用来看当前属性的信息熵的大小
            own_entropy = 0.0
            for value in unique_values:
                sub_data_set = split_data_set(data_set, i, value)  # 固定下标为i的特征后，得到单独的属性的子集（例如长相为帅的子集）
                prob = len(sub_data_set) / float(len(data_set))  # 获取长相为帅的属性值占所有行数的概率
                temp_entropy = calculate_shannon_entropy(sub_data_set)  # 即固定长相为其中某一个属性，比如帅时，它的信息熵是多少
                # 计算条件熵
                conditional_entropy += prob * temp_entropy
                # 分裂信息度量
                own_entropy -= prob * log(prob, 2)
            # 信息增益等于原始信息熵减去固定某一个特征后的条件熵，我们的目的是获取最大的信息增益
            info_gain = base_entropy - conditional_entropy
            if own_entropy == 0:  # 分裂属性度量为0
                continue
            info_gain_ratio = info_gain / own_entropy
            print(str(i) + '>>>>>>' + str(info_gain_ratio))
        if info_gain_ratio > best_info_gain_ratio:
            best_info_gain_ratio = info_gain_ratio
            best_feature = i

        # 处理连续值得特征，将之二元化
        if type(data_set[0][best_feature]).__name__ == 'float' or type(data_set[0][best_feature]).__name__ == 'int':
            best_split_value = best_split_dot.get(labels[best_feature])
            # for u in range(shape(data_set)[0]):
            for u in range(len(data_set)):
                if data_set[u][best_feature] <= best_split_value:
                    data_set[u][best_feature] = "<={value}".format(value=best_split_value)
                else:
                    data_set[u][best_feature] = ">{value}".format(value=best_split_value)
    return best_feature


# 测试决策树正确率
def testing(my_tree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify(my_tree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    print('myTree %d' % error)
    return float(error)


# 测试投票节点正确率
def testing_major(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    # print('major %d' % error)
    return float(error)


def get_count_for_classify(input_tree: dict, data_set, labels, result_count: list):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree.get(first_str)
    feature_index = labels.index(first_str)
    for k, v in second_dict.items():
        right_count = 0
        error_count = 0
        temp_labels = copy.deepcopy(labels)
        temp_labels.remove(first_str)
        sub_data_set = split_data_set(data_set, feature_index, k)
        if type(v).__name__ == 'dict':
            get_count_for_classify(v, sub_data_set, temp_labels, result_count)
        else:
            for each in sub_data_set:
                if str(each[-1]) == str(v):
                    right_count += 1
                else:
                    error_count += 1
            result_count.append([right_count, error_count])


def PEP_prune_branch(input_tree: dict, data_set, labels):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree.get(first_str)
    feature_index = labels.index(first_str)
    new_tree = {first_str: {}}
    labels.remove(first_str)
    for k, v in second_dict.items():
        if isinstance(v, dict):
            sub_data_set = split_data_set(data_set, feature_index, k)
            most_class = [ex[-1] for ex in sub_data_set]
            sub_tree_all_count = len(most_class)
            sub_tree_error_count = sub_tree_all_count - testing_major(majority_cnt(most_class), sub_data_set)
            result_count = []
            temp_labels = copy.deepcopy(labels)
            get_count_for_classify(v, sub_data_set, temp_labels, result_count)
            all_count = 0
            error_count = 0
            for right, error in result_count:
                all_count += right + error
                error_count += error
            if error_count == 0:  # 不存在错误
                new_tree[first_str].setdefault(k, v)
                continue
            sub_tree_error = error_count + len(result_count) * 0.5  # 整一棵子树的错误率，即剪枝之前的错误率
            leaf_error = sub_tree_error_count + 0.5  # 子树自己作为叶子节点的错误率
            p = sub_tree_error / all_count
            stand_wrong = sqrt(sub_tree_error * (1 - p))  # 标准差
            print('剪枝前 %s' % str(sub_tree_error -stand_wrong))
            print('剪枝后 %s' % str(leaf_error))
            if sub_tree_error - stand_wrong > leaf_error:  # 剪枝
                new_tree[first_str].setdefault(k, majority_cnt(most_class))
            else:
                # 继续内部剪枝
                sub_new_tree = PEP_prune_branch(v, sub_data_set, copy.deepcopy(labels))
                new_tree[first_str].setdefault(k, sub_new_tree)
        else:
            new_tree[first_str].setdefault(k, v)
    return new_tree

# # PEP 悲观剪枝
# def getCount(inputTree,dataSet,featLabels,count):
#     #global num
#     firstStr = inputTree.keys()[0]
#     secondDict = inputTree[firstStr]
#     featIndex = featLabels.index(firstStr)
#     #count=[]
#     for key in secondDict.keys():
#         rightcount = 0
#         wrongcount = 0
#         tempfeatLabels = featLabels[:]
#         subDataSet= split_data_set(dataSet, featIndex, key)
#         tempfeatLabels.remove(firstStr)
#         if type(secondDict[key]).__name__ == 'dict':
#             getCount(secondDict[key], subDataSet, tempfeatLabels, count)
#             #在这里加上剪枝的代码，可以实现自底向上的悲观剪枝
#         else:
#             for eachdata in subDataSet:
#                 if str(eachdata[-1]) == str(secondDict[key]):
#                     rightcount += 1
#                 else:
#                     wrongcount += 1
#             count.append([rightcount, wrongcount, secondDict[key]])
#             #num+=rightcount+wrongcount
#
#
# # PEP 悲观剪枝
# def cutBranch_uptodown(inputTree,dataSet,featLabels):    #自顶向下剪枝
#     firstStr=inputTree.keys()[0]
#     secondDict=inputTree[firstStr]
#     featIndex=featLabels.index(firstStr)
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=='dict':
#             tempfeatLabels=featLabels[:]
#         subDataSet= split_data_set(dataSet,featIndex,key)
#         tempfeatLabels.remove(firstStr)
#     tempcount=[]
#     getCount(secondDict[key],subDataSet,tempfeatLabels,tempcount)
#     print(tempcount)
#     #计算，并判断是否可以剪枝
#     #原误差率，显著因子取0.5
#     tempnum=0.0
#     wrongnum=0.0
#     old=0.0
#     #标准误差
#     standwrong=0.0
#     for var in tempcount:
#         tempnum+=var[0]+var[1]
#     wrongnum+=var[1]
#     old=float(wrongnum+0.5*len(tempcount))/float(tempnum)
#     standwrong=sqrt(tempnum*old*(1-old))
#     #假如剪枝
#     new=float(wrongnum+0.5)/float(tempnum)
#     if new<=old+standwrong and new >=old-standwrong:      #要确定新叶子结点的类别
#         #误判率最低的叶子节点的类为新叶子结点的类
#         #在count的每一个列表类型的元素里再加一个标记类别的元素。
#         wrongtemp=1.0
#         newtype=-1
#         for var in tempcount:
#             if float(var[1]+0.5)/float(var[0]+var[1])<wrongtemp:
#                 wrongtemp=float(var[1]+0.5)/float(var[0]+var[1])
#                 newtype=var[-1]
#         secondDict[key]=str(newtype)


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


def create_tree(data_set, labels, data_full, labels_full, test_data):
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
    best_feature_index = choose_best_feature(data_set, labels)
    # 获取label的名称
    best_label = labels[best_feature_index]
    # unique_list_full = None
    labels_copy = copy.deepcopy(labels)  # 用来剪枝
    '''
    刚开始很奇怪为什么要加一个uniqueValFull，后来思考下觉得应该是在某次划分，比如在根节点划分纹理的时候，将数据分成了清晰、模糊、稍糊三块
    ，假设之后在模糊这一子数据集中，下一划分属性是触感，而这个数据集中只有软粘属性的西瓜，这样建立的决策树在当前节点划分时就只有软粘这一属性了，
    事实上训练样本中还有硬滑这一属性，这样就造成了树的缺失，因此用到uniqueValFull之后就能将训练样本中有的属性值都囊括。
    如果在某个分支每找到一个属性，就在其中去掉一个，最后如果还有剩余的根据父节点投票决定。
    但是即便这样，如果训练集中没有出现触感属性值为“一般”的西瓜，但是分类时候遇到这样的测试样本，那么应该用父节点的多数类作为预测结果输出。
    '''
    # if type(data_set[0][best_feature_index]).__name__ == 'str':
    #     current_label_index = labels_full.index(labels[best_feature_index])
    #     feature_values_full = [example[current_label_index] for example in data_full]
    #     unique_list_full = set(feature_values_full)
    # 初始化tree
    my_tree = {best_label: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[best_feature_index])
    # 取出最优列，然后它的branch做分类
    feature_values = [example[best_feature_index] for example in data_set]  # 假设最优列是长相
    unique_list = set(feature_values)
    for unique in unique_list:
        # if type(data_set[0][best_feature_index]).__name__ == 'str' and \
        #         ('>' in type(data_set[0][best_feature_index]).__name__ or
        #          '<=' in type(data_set[0][best_feature_index]).__name__):
        #     unique_list_full.remove(unique)
        # 求出剩余的label
        sub_labels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()， 分别创建子树
        # 即分割长相这个最优列后寻找新的最优列
        my_tree[best_label][unique] = create_tree(split_data_set(data_set, best_feature_index, unique), sub_labels,
                                                  data_full, labels_full, split_data_set(test_data, best_feature_index,
                                                                                         unique))
    # if type(data_set[0][best_feature_index]).__name__ == 'str':
    #     for v in unique_list_full:
    #         my_tree[best_label][v] = majority_cnt(class_list)
    # 如果测试的错误率大于
    # if testing(my_tree, test_data, labels_copy) > testing_major(majority_cnt(class_list), test_data):
    #     return majority_cnt(class_list)
    # print(my_tree)
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
    if type(key).__name__ == 'float' or type(key).__name__ == 'int':
        keys = list(secondDict.keys())
        if len(keys) > 0:
            keys = keys[0]
        split_key = str(keys).replace('>', '').replace('<=', '')
        if float(key) <= float(split_key):
            key = "<={value}".format(value=split_key)
        else:
            key = ">{value}".format(value=split_key)
    valueOfFeat = secondDict[key]
    # print('+++' + str(firstStr) + 'xxx' + str(secondDict) + '---' + str(key) + '>>>' + str(valueOfFeat))
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def fishTest(is_flush=False):
    # 1.创建数据和结果标签
    myDat, labels = create_data_set()
    train_data = myDat[:]
    test_data = myDat[:]
    import copy
    if not is_flush:
        myTree = grabTree('./simple.m')
    else:
        myTree = create_tree(train_data, copy.deepcopy(labels), myDat, copy.deepcopy(labels), test_data)
        print(myTree)
        storeTree(myTree, './simple.m')
    myTree = PEP_prune_branch(myTree, myDat, copy.deepcopy(labels))
    k = 0
    for item in myDat:
        if classify(myTree, labels[:], item) == item[-1]:
            k += 1
    print(float(k) / len(myDat))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    # print(classify(myTree, labels, ['丑', '豪', '高', 34]))

    # 画图可视化展现
    createPlot(myTree)


def storeTree(inputTree, filename):
    import pickle
    # -------------- 第一种方法 start --------------
    # fw = open(filename, 'w')
    # pickle.dump(inputTree, fw)
    # fw.close()
    # -------------- 第一种方法 end --------------

    # -------------- 第二种方法 start --------------
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
    # -------------- 第二种方法 start --------------


def grabTree(filename):
    import pickle
    with open(filename, 'rb') as fr:
        return pickle.load(fr)


if __name__ == '__main__':
    fishTest()
    # lenses_tree = contact_lensesTest()
    # storeTree(lenses_tree, 'lenses_tree.m')
    # createPlot(grabTree('lenses_tree.m'))