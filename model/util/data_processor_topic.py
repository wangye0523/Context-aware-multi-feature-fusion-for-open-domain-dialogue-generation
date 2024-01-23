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
            str_posts, str_responses, str_responses_act, str_responses_emotion, str_keyword = [], [], [], [], []  # post和response的str表示
            for item in batch_data:
                str_posts.append(item['post'])
                str_responses.append(item['response'])
                str_keyword.append(item['KeyWord'])
                str_responses_act.append(item['response_label_act'])
                str_responses_emotion.append(item['response_label_emotion'])

            id_posts, id_responses, id_responses_act, id_responses_emotion, id_keywords = [], [], [], [], []
            len_posts, len_responses, len_responses_act, len_responses_emotion, len_keywords = [], [], [], [], []
            for post in str_posts:  # post从str2index并统计长度
                id_post, len_post = self.sp.word2index(post)
                id_posts.append(id_post)
                len_posts.append(len_post)

            for response in str_responses:  # response从str2index并统计长度
                id_response, len_response = self.sp.word2index(response)
                id_responses.append(id_response)
                len_responses.append(len_response)

            for act in str_responses_act:
                id_response_act, len_response_act = self.sp.word2index(act)
                id_responses_act.append(id_response_act)
                len_responses_act.append(len_response_act)

            for emotion in str_responses_emotion:
                id_response_emotion, len_response_emotion = self.sp.word2index(emotion)
                id_responses_emotion.append(id_response_emotion)
                len_responses_emotion.append(len_response_emotion)

            for keyword in str_keyword:
                id_keyword, len_keyword = self.sp.word2index(keyword)
                id_keywords.append(id_keyword)
                len_keywords.append(len_keyword)

            len_posts = [l+2 for l in len_posts]  # 加上start和end后的长度
            len_responses = [l+2 for l in len_responses]
            len_keywords = [l for l in len_keywords]

            maxlen_post = max(len_posts)
            maxlen_response = max(len_responses)
            maxlen_keyword = max(len_keywords)

            pad_id_posts = [self.sp.pad_sentence(p, maxlen_post) for p in id_posts]  # 补齐长度
            pad_id_responses = [self.sp.pad_sentence(r, maxlen_response) for r in id_responses]
            pad_id_keywords = [self.sp.pad_sentence_keyword(k, maxlen_keyword) for k in id_keywords]

            new_batch_data = {'str_posts': str_posts,
                              'str_responses': str_responses,
                              'str_keywords':str_keyword,
                              'posts': pad_id_posts,
                              'responses': pad_id_responses,
                              'keywords': pad_id_keywords,
                              'len_posts': len_posts,
                              'len_responses': len_responses,
                              'len_keywords': len_keywords,
                              'str_responses_act': str_responses_act,
                              'len_responses_act': len_responses_act,
                              'responses_act': id_responses_act,
                              'str_responses_emotion': str_responses_emotion,
                              'len_responses_emotion': len_responses_emotion,
                              'responses_emotion': id_responses_emotion,
                              }

            yield new_batch_data
