import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np
from model.Embedding import Embedding
from model.Encoder import Encoder
from model.PriorNet import PriorNet
from model.RecognizeNet import RecognizeNet
from model.Decoder_attention import Decoder
from model.PrepareState import PrepareState
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        # 定义嵌入层
        self.embedding = Embedding(config.num_vocab,  # 词汇表大小
                                   config.embedding_size,  # 嵌入层维度
                                   config.pad_id,  # pad_id
                                   config.dropout)

        # post编码器
        self.post_encoder = Encoder(config.post_encoder_cell_type,  # rnn类型
                                    config.embedding_size,  # 输入维度
                                    config.post_encoder_output_size,  # 输出维度
                                    config.post_encoder_num_layers,  # rnn层数
                                    config.post_encoder_bidirectional,  # 是否双向
                                    config.dropout)  # dropout概率

        # response编码器
        self.response_encoder = Encoder(config.response_encoder_cell_type,
                                        config.embedding_size,  # 输入维度
                                        config.response_encoder_output_size,  # 输出维度
                                        config.response_encoder_num_layers,  # rnn层数
                                        config.response_encoder_bidirectional,  # 是否双向
                                        config.dropout)  # dropout概率

        # keyword编码器
        self.keyword_encoder = Encoder(config.post_encoder_cell_type,  # rnn类型
                                    config.embedding_size,  # 输入维度
                                    config.post_encoder_output_size,  # 输出维度
                                    config.post_encoder_num_layers,  # rnn层数
                                    config.post_encoder_bidirectional,  # 是否双向
                                    config.dropout)  # dropout概率

        # category 编码器
        self.category_encoder = Encoder(config.post_encoder_cell_type,  # rnn类型
                                       config.embedding_size,  # 输入维度
                                       config.post_encoder_output_size,  # 输出维度
                                       config.post_encoder_num_layers,  # rnn层数
                                       config.post_encoder_bidirectional,  # 是否双向
                                       config.dropout)  # dropout概率

        self.atten_category_keywords = nn.Sequential()
        self.atten_category_keywords.add_module("1" ,nn.Linear(config.decoder_output_size * 2, 250))
        self.atten_category_keywords.add_module("2", nn.Tanh())
        self.atten_category_keywords.add_module("3", nn.Linear(250, config.decoder_output_size))

        # 先验网络
        self.prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                  config.latent_size,  # 潜变量维度
                                  config.dims_prior)  # 隐藏层维度

        # 识别网络
        self.recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                          config.response_encoder_output_size,  # response输入维度
                                          config.latent_size,  # 潜变量维度
                                          config.dims_recognize)  # 隐藏层维度

        self.post_category_keywords = nn.Sequential()
        self.post_category_keywords.add_module("4", nn.Linear(config.decoder_output_size * 2, 250))
        self.post_category_keywords.add_module("5", nn.Tanh())
        self.post_category_keywords.add_module("6", nn.Linear(250, config.decoder_output_size))

        # 初始化解码器状态
        self.prepare_state = PrepareState(config.post_encoder_output_size+config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        # 解码器
        self.decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率


        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

    def forward(self, inputs, word2vec ,inference=False, inpre=False, ingau=False, max_len=60, gpu=True):
        if not inference:  # 训练
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            len_decoder = id_responses.size(1) - 1
            id_keywords = inputs["keywords"]
            len_keywords = inputs["len_keywords"]
            id_category = inputs["category"]
            len_category = inputs["len_category"]

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = word2vec.embedding(id_responses)  # [batch, seq, embed_size]
            embed_keywords = word2vec.embedding(id_keywords)
            embed_category = word2vec.embedding(id_category)
            # state: [layers, batch, dim]
            encoder_output, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            encoder_output_responses, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            keywords_output, state_keywords = self.keyword_encoder(embed_keywords.transpose(0, 1), len_keywords)
            category_output, state_category = self.category_encoder(embed_category.transpose(0, 1), len_category)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[1]
            if isinstance(state_keywords, tuple):
                state_keywords = state_keywords[1]
            if isinstance(state_category, tuple):
                state_category = state_category[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]
            keywords_state = state_keywords[-1, :, :]  # [batch, dim]
            category_state = state_category[-1, :, :]  # [batch, dim]

            category_keyword = self.atten_category_keywords(torch.cat([keywords_state, category_state], 1))

            x = self.post_category_keywords(torch.cat([x, category_keyword],1))
            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            # p(z|x,y)
            mu, logvar = self.recognize_net(x, y)  # [batch, latent]

            z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]




            # 解码器的输入为回复去掉end_id
            decoder_inputs = embed_responses[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            decoder_inputs = decoder_inputs.split([1] * len_decoder, 0)  # 解码器每一步的输入 seq-1个[1, batch, embed_size]
            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]

            outputs = []
            for idx in range(len_decoder):
                if idx == 0:
                    state = first_state  # 解码器初始状态
                decoder_input = decoder_inputs[idx]  # 当前时间步输入 [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(decoder_input, state, encoder_output)
                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]


            return output_vocab, _mu, _logvar, mu, logvar
        else:  # 测试
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            id_keywords = inputs["keywords"]
            len_keywords = inputs["len_keywords"]
            id_category = inputs["category"]
            len_category = inputs["len_category"]
            batch_size = id_posts.size(0)

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_keywords = word2vec.embedding(id_keywords)
            embed_category = word2vec.embedding(id_category)
            # state: [layers, batch, dim]
            encoder_output, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            keywords_output, state_keywords = self.keyword_encoder(embed_keywords.transpose(0, 1), len_keywords)
            category_output, state_category = self.category_encoder(embed_category.transpose(0, 1), len_category)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_keywords, tuple):
                state_keywords = state_keywords[1]
            if isinstance(state_category, tuple):
                state_category = state_category[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            keywords_state = state_keywords[-1, :, :]  # [batch, dim]
            category_state = state_category[-1, :, :]  # [batch, dim]

            category_keyword = self.atten_category_keywords(torch.cat([keywords_state, category_state], 1))
            x = self.post_category_keywords(torch.cat([x, category_keyword], 1));

            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]

            z = _mu + (0.5 * _logvar).exp() * sampled_latents  # [batch, latent]

            first_state = self.prepare_state(torch.cat([z,x], 1))
            done = torch.tensor([0] * batch_size).bool()
            first_input_id = (torch.ones((1, batch_size)) * self.config.start_id).long()
            if gpu:
                done = done.cuda()
                first_input_id = first_input_id.cuda()

            outputs = []
            for idx in range(max_len):
                if idx == 0:  # 第一个时间步
                    state = first_state  # 解码器初始状态
                    decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                else:
                    decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layers, batch, dim_out]
                output, state = self.decoder(decoder_input, state ,encoder_output)
                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq, num_vocab]

            return output_vocab

    def print_parameters(self):
        r""" 统计参数 """
        total_num = 0  # 参数总数
        for param in self.parameters():
            num = 1
            if param.requires_grad:
                size = param.size()
                for dim in size:
                    num *= dim
            total_num += num
        print(f"参数总数: {total_num}")

    def save_model(self, epoch, global_step, path):
        r""" 保存模型 """
        torch.save({'embedding': self.embedding.state_dict(),
                    'post_encoder': self.post_encoder.state_dict(),
                    'category_encoder': self.category_encoder.state_dict(),
                    'keyword_encoder': self.keyword_encoder.state_dict(),
                    'prepare_state': self.prepare_state.state_dict(),
                    'atten_category_keywords': self.atten_category_keywords.state_dict(),
                    'prior_net': self.prior_net.state_dict(),
                    'recognize_net': self.recognize_net.state_dict(),
                    'post_category_keywords': self.post_category_keywords.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'projector': self.projector.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.post_encoder.load_state_dict(checkpoint['post_encoder'])
        self.category_encoder.load_state_dict(checkpoint['category_encoder'])
        self.atten_category_keywords.load_state_dict(checkpoint['atten_category_keywords'])
        self.keyword_encoder.load_state_dict(checkpoint['keyword_encoder'])
        self.post_category_keywords.load_state_dict(checkpoint['post_category_keywords'])
        self.prior_net.load_state_dict(checkpoint['prior_net'])
        self.recognize_net.load_state_dict(checkpoint['recognize_net'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        return epoch, global_step

    def prepare_feed_data(self,data, inference=False):
        len_labels = torch.tensor([l - 1 for l in data['len_responses']]).long()  # [batch] 标签没有start_id，长度-1
        masks = (1 - F.one_hot(len_labels, len_labels.max() + 1).cumsum(1))[:, :-1]  # [batch, len_decoder]
        batch_size = masks.size(0)

        if not inference:  # 训练时的输入
            feed_data = {'posts': torch.tensor(data['posts']).long(),  # [batch, len_encoder]
                         'len_posts': torch.tensor(data['len_posts']).long(),  # [batch]
                         'responses': torch.tensor(data['responses']).long(),  # [batch, len_decoder]
                         'len_responses': torch.tensor(data['len_responses']).long(),  # [batch]
                         'sampled_latents': torch.randn((batch_size, self.config.latent_size)),  # [batch, latent_size]
                         'masks': masks.float(),
                         'responses_act': torch.tensor(data['responses_act'])}  # [batch, len_decoder]
        else:  # 测试时的输入
            feed_data = {'posts': torch.tensor(data['posts']).long(),
                         'len_posts': torch.tensor(data['len_posts']).long(),
                         'sampled_latents': torch.randn((batch_size, self.config.latent_size))}

        if True:  # 将数据转移到gpu上
            for key, value in feed_data.items():
                feed_data[key] = value.cuda()

        return feed_data

