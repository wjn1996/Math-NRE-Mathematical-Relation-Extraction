# Math-NRE-Mathematical-Relation-Extraction
Math-NRE：中学数学知识抽取——关系抽取

# Introduction

介绍：关系抽取是构建知识图谱的主要过程，为了构建中学数学知识地图，博主已经先提供了中学数学知识点实体识别算法及模型：[Mathematical-Knowledge-Entity-Recognition](https://github.com/wjn1996/Mathematical-Knowledge-Entity-Recognition) ，现今将开源关系抽取模型。关系抽取目前选择TextCNN和基于注意力机制的LSTM两种Baseline模型。

# Algorithm

算法模型讲解：可参考博主的CSDN博客：[基于监督学习和远程监督的神经关系抽取](https://blog.csdn.net/qq_36426650/article/details/103219167)
- 词向量：使用word2vec或golve训练中文词向量，本文使用的是字符级别向量，预训练的向量下载地址：[word2vec](https://download.csdn.net/download/qq_36426650/11828651)、[Glove](https://download.csdn.net/download/qq_36426650/11828655)

- TextCNN：输入预训练词向量，使用一层2维度卷积神经网络，并进行最大池化。输出部分则为一层全连接网络和softmax。
- Att-BiLSTM：输入预训练词向量，使用一层LSTM，输出部分使用注意力机制和全连接网络。


# Experiments

实验说明：实验除了对比不同词向量、不同模型关系抽取效果外，还研究句子各个部分对关系抽取的影响。其中包括实体间距（句子中两个实体之间的间隔字符数）和实体外围（头实体左侧词的个数、尾实体右侧词的个数）。例如对于句子：“如果两个正整数的最大公约数是1，则它们是互素的。”，若实体对为（最大公约数，互素），则它们的实体间距为7，当实体外围设置为0时，即“最大公约数是1，则它们是互素”，若实体外围为2，则为“数的最大公约数是1，则它们是互素的。”。因此制作相关的数据集。注：最大实体间距是指这个数据集中所有的句子中实体对之间的字符数最大个数。

 - 数据集：data目录：
re_50_k_j_train_data:表示实体间距最大为50，间距最小为k，padding为j
k取值为0 5 10 15 20；j取值为0 2 4 6 8
 - 训练：例如使用word2vec来训练初中数学（junior_math），最大实体间距为50，实体外围为0，则执行：
python3 train.py --embedding_type=word2vec --train_path=./data/junior_math/re_50_0_0_train_data
其中：embedding_type可取空（random）、word2vec、glove和gwe 
train_path为训练的数据集路径
gpu：指定gpu
 - 测试：使用测试集进行测试
python3 eval.py --test_path "data/junior_math/re_50_0_0_test_data" --checkpoint_dir "runs/1551410876/checkpoints" --model_dir "runs/1551410876/"
test_path：测试集路径
checkpoint_dir：checkpoint路径
model_dir：模型保存路径

# Result
实验结果表明：
- （1）glove词向量对关系抽取的效果最好；
- （2）Att-BiLSTM比CNN效果好；
- （3）当实体外围一定时，实体最小间距不宜太小也不宜太大；当实体最小间距一定时，实体外围为0时最优。


