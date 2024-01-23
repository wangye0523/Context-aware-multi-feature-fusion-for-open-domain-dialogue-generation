from model.util.data_iterator import DataIterator
import random


class DataProcessor(object):
    r""" 实现数据的预处理 """
    def __init__(self, data, batch_size, sp, shuffle=True):
        self.sp = sp
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_batch_data(self):
        r""" 输出一个batch预处理的样本 """
        if self.shuffle:
            random.shuffle(self.data)
        it = DataIterator(self.data, self.batch_size)

        for batch_data in it.get_batch_data():
            str_sentences = []
            for item in batch_data:
                str_sentences.append(item['post']+['<SOS>']+item['response'])

            id_sentences = []
            len_sentences = []
            for str_sentence in str_sentences:  # post从str2index并统计长度
                id_sentence, len_sentence = self.sp.word2index(str_sentence)
                id_sentences.append(id_sentence)
                len_sentences.append(len_sentence)

            len_sentences = [l+2 for l in len_sentences]  # 加上start和end后的长度

            maxlen_sentence = max(len_sentences)

            pad_id_sentences = [self.sp.pad_sentence(p, maxlen_sentence) for p in id_sentences]  # 补齐长度

            new_batch_data = {'str_sentences': str_sentences,
                              'id_sentences': pad_id_sentences,
                              'len_sentences': len_sentences,
                              }

            yield new_batch_data
