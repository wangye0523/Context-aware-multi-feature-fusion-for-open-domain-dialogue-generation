from gensim.models.word2vec import Word2Vec
import torch
# # #数据的读入
with open('data/raw/vocab.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
vocab = []
for line in lines:
    vocab.append(line.replace('\n', ''))

# file = open('data/raw/dialogues_text.txt')
# ops = []
# for line in file.readlines():
#     curLine=line.strip().replace('__eou__','').split(" ")
#     # curLine = [i for i in curLine if i not in vocab i = '<UNK>']
#     data_line = []
#     for i in curLine:
#         if i not in vocab:
#             data_line.append('<UNK>')
#         else:
#             data_line.append(i)
#     # floatLine=list(map(float,curLine)) #这里使用的是map函数直接把数据转化成为float类型
#     ops.append(['<SOS>']+data_line+['<PAD>', '<EOS>'])
# #模型的训练

#
#
#
# w2vModel = Word2Vec(vector_size=300, min_count=1)
# # 加载词表
# w2vModel.build_vocab(ops)
# # 训练
# w2vModel.train(ops, total_examples=w2vModel.corpus_count, epochs=100)
# print("相似度计算:", w2vModel.wv.similarity(['I']))
# 保存方法一
# w2vModel.save('w2vModel.model')
class Word2Vec_emb():
    def __init__(self):
        self.vocab = vocab
        self.w2vModel = Word2Vec.load('w2vModel.model')

    def embedding(self,batch_data):
        emb_x, emb_y = [], []
        for x in batch_data:
            for y in x:
                emb_y.append(torch.tensor(self.w2vModel.wv[self.vocab[y.item()]]))
            emb_x.append(torch.stack(emb_y, dim=0))
            emb_y = []
        embedding_ = torch.stack(emb_x, dim=0)
        embedding_.requires_grad = True
        return embedding_.cuda()



# from model.Embedding import Embedding
# embedding = Embedding(10000, 300, 0, 0.5)
# print(embedding(batch_data).size())
# print("相似度计算:", w2vModel.wv.similarity('<EOS>', '<SOS>'))
# for word in w2vModel.wv.index_to_key:
#     if word == '<PAD>':
#         print(word)



