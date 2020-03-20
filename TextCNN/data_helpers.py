import numpy as np
import pandas as pd
import nltk
import re
import utils



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    # return [x_text, y]

    #edit by wjn
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
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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
