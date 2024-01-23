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
            str_posts, str_responses, str_responses_act, str_responses_emotion = [], [], [], []  # post和response的str表示
            for item in batch_data:
                str_posts.append(item['commissive'])
                str_responses.append(item['post'])
                if(item['post_label_act'][0] == '0'):
                    str_responses_act.append([0])
                elif(item['post_label_act'][0] == '1'):
                    str_responses_act.append([1])
                elif (item['post_label_act'][0] == '2'):
                    str_responses_act.append([2])
                else:
                    str_responses_act.append([3])

                if (item['post_label_act'][0] == '0'):
                    str_responses_emotion.append([0])
                elif (item['post_label_act'][0] == '1'):
                    str_responses_emotion.append([1])
                else:
                    str_responses_emotion.append([2])

            id_posts, id_responses, id_responses_act, id_responses_emotion = [], [], [], []
            len_posts, len_responses, len_responses_act, len_responses_emotion = [], [], [], []
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
            len_posts = [l+2 for l in len_posts]  # 加上start和end后的长度
            len_responses = [l+2 for l in len_responses]

            maxlen_post = max(len_posts)
            maxlen_response = max(len_responses)

            pad_id_posts = [self.sp.pad_sentence(p, maxlen_post) for p in id_posts]  # 补齐长度
            pad_id_responses = [self.sp.pad_sentence(r, maxlen_response) for r in id_responses]

            new_batch_data = {'str_posts': str_posts,
                              'str_responses': str_responses,
                              'str_keywords':str_responses,
                              'posts': pad_id_posts,
                              'responses': pad_id_responses,
                              'len_posts': len_posts,
                              'len_responses': len_responses,
                              'str_post_act': str_responses_act,
                              'len_post_act': len_responses_act,
                              'post_act': str_responses_act,
                              'str_post_emotion': str_responses_emotion,
                              'len_post_emotion': len_responses_emotion,
                              'post_emotion': str_responses_emotion,
                              }

            yield new_batch_data
