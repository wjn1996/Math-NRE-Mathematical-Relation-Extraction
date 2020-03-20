import numpy as np
import pandas as pd
import nltk
import re

import utils
from configure import FLAGS

#将单词表示进行修整
def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace('\xa0','')

    return text.strip()
'''
#加载task8数据集并做预处理，返回句子列表以及对应的标签独热编码
def load_data_and_labels(path):
    data = []#加载处理后的数据集
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0#最长句子的长度
    for idx in range(0, len(lines), 4):#task8数据集每四行为一个样本
        id = lines[idx].split("\t")[0]#提取编号
        relation = lines[idx + 1]#提取关系

        sentence = lines[idx].split("\t")[1][1:-1]#提取去除首尾引号的原句字符串
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)#分词
        if max_sentence_length < len(tokens):#获取整个样本中最长句子的长度
            max_sentence_length = len(tokens)
        sentence = " ".join(tokens)#列表转字符串并以空格分割每一个元素

        data.append([id, sentence, relation])#按照一定规则生成样本列表

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [utils.class2label[r] for r in df['relation']]
    ###上两句产生的结果是(样例)：
    ###print(df)
    ###   id sentence relation  label
    ###0   1    43dad        a      1
    ###1   2    sjaon        c      3
    ###
    # Text Data，例如上例中为['43dad', 'sjaon']
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()#y的值转化为array，上例中为array([1, 3], dtype=int64)
    labels_count = np.unique(labels_flat).shape[0]#标签计数

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    #将标签编号转化为独热编码
    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    #返回句子列表以及对应的标签独热编码
    return x_text, labels
'''

#add ny wjn 
#加载训练集
def load_data_and_labels(path):
    max_sentence_length = 0#最长句子的长度
    data=[]
    with open(path,'r',encoding="UTF-8-sig") as f:
        for ei,i in enumerate(f.readlines()):
            sentence,entity1,entity2,relation = i.strip().split(' ')
            sentence = ' '.join([x for x in sentence])
            tokens = sentence.split(' ')
            if max_sentence_length < len(tokens):#获取整个样本中最长句子的长度
                max_sentence_length = len(tokens)
            data.append([ei+1, sentence, relation])#按照一定规则生成样本列表
    print('data=',data[0])
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [utils.class2label[r] for r in df['relation']]
    ###上两句产生的结果是(样例)：
    ###print(df)
    ###   id sentence relation  label
    ###0   1    43dad        a      1
    ###1   2    sjaon        c      3
    ###
    # Text Data，例如上例中为['43dad', 'sjaon']
    x_text = df['sentence'].tolist()
    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()#y的值转化为array，上例中为array([1, 3], dtype=int64)
    labels_count = np.unique(labels_flat).shape[0]#标签计数
    print('labels_count=',labels_count)
    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    #将标签编号转化为独热编码
    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    #返回句子列表以及对应的标签独热编码
    return x_text, labels            

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    trainFile = 'data\\junior_math\\re_50_5_4_train_data'
    testFile = 'data\\junior_math\\re_50_5_4_test_data'

    print(load_data_and_labels(trainFile))
