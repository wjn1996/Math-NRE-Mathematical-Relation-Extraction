import numpy as np
import gensim


# class2label = {'Other': 0,
#                'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
#                'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
#                'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
#                'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
#                'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
#                'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
#                'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
#                'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
#                'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

# label2class = {0: 'Other',
#                1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
#                3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
#                5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
#                7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
#                9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
#                11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
#                13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
#                15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
#                17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


class2label = {'other': 0,
               'rely': 1, 'b-rely': 2,
               'belg': 3, 'b-belg': 4,
               'syno': 5, 'anto': 6,
               'simi': 7, 'attr': 8,
               'b-attr': 9, 'appo': 10}

label2class = {0: 'other',
               1: 'rely', 2: 'b-rely',
               3: 'belg', 4: 'b-belg',
               5: 'syno', 6: 'anto',
               7: 'simi', 8: 'attr',
               9: 'b-attr', 10: 'appo'}


#加载训练好的模型词向量,包括三类词向量w2v，glove，GWE
def load_embeddings(embedding_dim, vocab, embedding_type):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    # print("Load glove file {0}".format(embedding_path))
    embedding_dir = './embeddings/'
    if embedding_type == "glove":
        f = open(embedding_dir + 'wiki.zh.glove.Mode', 'r', encoding='utf8')
        for line in f:
            splitLine = line.split(' ')
            word = splitLine[0]
            embedding = np.asarray(splitLine[1:], dtype='float32')
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = embedding
    elif embedding_type == "word2vec":
        model = gensim.models.Word2Vec.load(embedding_dir + 'wiki.zh.w2v.Mode')
        allwords = model.wv.vocab
        for word in allwords:
            embedding = np.asarray(model[word], dtype='float32')
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = embedding
    elif embedding_type == "gwe":
        with open(embedding_dir + 'wiki.zh.GWE.mode','r',encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                splitLine = line.split(' ')
                word = splitLine[0]
                embedding = np.asarray(splitLine[1:301], dtype='float32')
                idx = vocab.vocabulary_.get(word)
                if idx != 0:
                    initW[idx] = embedding
    return initW
# if __name__ == '__main__':
#   load_embeddings(300,)