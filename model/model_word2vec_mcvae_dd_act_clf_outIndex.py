import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np
from model.Embedding import Embedding
from model.Encoder import Encoder
from model.PriorNet import PriorNet
from model.RecognizeNet import RecognizeNet
from model.Decoder import Decoder
from model.PrepareState import PrepareState
import torch.nn.functional as F

act_num = 4

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
                                    config.dropout) # dropout概率

        # response编码器
        self.response_encoder = Encoder(config.response_encoder_cell_type,
                                        config.embedding_size,  # 输入维度
                                        config.response_encoder_output_size,  # 输出维度
                                        config.response_encoder_num_layers,  # rnn层数
                                        config.response_encoder_bidirectional,  # 是否双向
                                        config.dropout)  # dropout概率

        # classification
        self.classification = nn.Sequential(
            nn.Linear(config.post_encoder_output_size, act_num),
            nn.Softmax(-1)
        )

        # 先验网络
        self.prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                  config.latent_size,  # 潜变量维度
                                  config.dims_prior)  # 隐藏层维度

        # 识别网络
        self.recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                          config.response_encoder_output_size,  # response输入维度
                                          config.latent_size,  # 潜变量维度
                                          config.dims_recognize)  # 隐藏层维度

        # inform先验网络
        self.inform_prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                  config.latent_size,  # 潜变量维度
                                  config.dims_prior)  # 隐藏层维度

        # inform识别网络
        self.inform_recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                          config.response_encoder_output_size,  # response输入维度
                                          config.latent_size,  # 潜变量维度
                                          config.dims_recognize)  # 隐藏层维度

        # question先验网络
        self.question_prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                         config.latent_size,  # 潜变量维度
                                         config.dims_prior)  # 隐藏层维度

        # question识别网络
        self.question_recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                                 config.response_encoder_output_size,  # response输入维度
                                                 config.latent_size,  # 潜变量维度
                                                 config.dims_recognize)  # 隐藏层维度

        # directive先验网络
        self.directive_prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                           config.latent_size,  # 潜变量维度
                                           config.dims_prior)  # 隐藏层维度

        # directive识别网络
        self.directive_recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                                   config.response_encoder_output_size,  # response输入维度
                                                   config.latent_size,  # 潜变量维度
                                                   config.dims_recognize)  # 隐藏层维度

        # commissive先验网络
        self.commissive_prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                            config.latent_size,  # 潜变量维度
                                            config.dims_prior)  # 隐藏层维度

        # commisive识别网络
        self.commissive_recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                                    config.response_encoder_output_size,  # response输入维度
                                                    config.latent_size,  # 潜变量维度
                                                    config.dims_recognize)  # 隐藏层维度

        # 多语义先验网络
        self.ms_prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                             config.latent_size,  # 潜变量维度
                                             config.dims_prior)  # 隐藏层维度

        # 多语义识别网络
        self.ms_recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                                     config.response_encoder_output_size,  # response输入维度
                                                     config.latent_size,  # 潜变量维度
                                                     config.dims_recognize)  # 隐藏层维度

        # 初始化解码器状态
        self.prepare_state = PrepareState(config.post_encoder_output_size+config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        # 初始化解码器状态
        self.inform_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)
        # 初始化解码器状态
        self.question_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)
        # 初始化解码器状态
        self.directive_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)
        # 初始化解码器状态
        self.commissive_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                          config.decoder_cell_type,
                                          config.decoder_output_size,
                                          config.decoder_num_layers)

        # 初始化解码器状态
        self.ms_prepare_state = PrepareState(config.post_encoder_output_size + config.latent_size,
                                                     config.decoder_cell_type,
                                                     config.decoder_output_size,
                                                     config.decoder_num_layers)

        self.input_cat_pre = nn.Linear(config.embedding_size+config.decoder_output_size, config.decoder_output_size)

        # 解码器
        self.decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        #inform  decoder
        self.inform_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        #question  decoder
        self.question_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        #directive decoder
        self.directive_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embedding_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layers,  # rnn层数
                               config.dropout)  # dropout概率
        # commissive  decoder
        self.commissive_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                                        config.embedding_size,  # 输入维度
                                        config.decoder_output_size,  # 输出维度
                                        config.decoder_num_layers,  # rnn层数
                                        config.dropout)  # dropout概率
        # commissive  decoder
        self.ms_decoder = Decoder(config.decoder_cell_type,  # rnn类型
                                          config.embedding_size,  # 输入维度
                                          config.decoder_output_size,  # 输出维度
                                          config.decoder_num_layers,  # rnn层数
                                          config.dropout)  # dropout概率


        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )
        self.projector_inform = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_question = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_directive = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_commssive = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        self.projector_ms = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

    def forward(self, inputs, word2vec, inference=False, ms = False ,inpre=False, ingau=False, max_len=60, gpu=True):
        if inpre:
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            len_decoder = id_responses.size(1) - 1

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = word2vec.embedding(id_responses)  # [batch, seq, embed_size]
            # state: [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]

            classify_result = self.classification(x) #[batch, cat_num]
            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            # p(z|x,y)
            mu, logvar = self.recognize_net(x, y)  # [batch, latent]
            self.mu1 = mu
            self.sigma1 = logvar
            # 重参数化
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
                output, state = self.decoder(decoder_input, state)
                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]

            return output_vocab, _mu, _logvar, mu, logvar, classify_result
        if ms:
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            id_catgory = torch.tensor([0,1,2,3]).long().repeat(sampled_latents.size(0),1).cuda()#[batch,cat_num]
            len_decoder = id_responses.size(1) - 1

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = word2vec.embedding(id_responses)  # [batch, seq, embed_size]
            embed_catgory = word2vec.embedding(id_catgory) #[batch, cat_num, embed_size]
            # state: [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]

            classify_result = self.classification(x)  # [batch, cat_num]
            input_cat = torch.bmm(embed_catgory.transpose(1, 2), torch.unsqueeze(classify_result, dim=2)).transpose(1,2).transpose(0,1)#[1, batch, embed_size]
            # p(z|x)
            _mu, _logvar = self.ms_prior_net(x)  # [batch, latent]
            # p(z|x,y)
            mu, logvar = self.ms_recognize_net(x, y)  # [batch, latent]
            self.mu1 = mu
            self.sigma1 = logvar
            # 重参数化
            z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]

            # 解码器的输入为回复去掉end_id
            decoder_inputs = embed_responses[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            decoder_inputs = decoder_inputs.split([1] * len_decoder, 0)  # 解码器每一步的输入 seq-1个[1, batch, embed_size]
            first_state = self.ms_prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]
            outputs = []
            for idx in range(len_decoder):
                if idx == 0:
                    state = first_state  # 解码器初始状态
                decoder_input = decoder_inputs[idx]  # 当前时间步输入 [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.ms_decoder(decoder_input, state)

                output = self.input_cat_pre(torch.cat((output,input_cat),2))
                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]
            output_vocab = self.projector_ms(outputs)  # [batch, seq-1, num_vocab]

            return output_vocab, _mu, _logvar, mu, logvar, classify_result
        elif not inference:  # 训练
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            len_decoder = id_responses.size(1) - 1

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = word2vec.embedding(id_responses)  # [batch, seq, embed_size]
            # state: [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]
            classify_result = self.classification(x)

            if ingau == 0:
                _mu, _logvar = self.inform_prior_net(x)  # [batch, latent]
                mu, logvar = self.inform_recognize_net(x, y)  # [batch, latent]
                z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
            elif ingau == 1:
                _mu, _logvar = self.question_prior_net(x)  # [batch, latent]
                mu, logvar = self.question_recognize_net(x, y)  # [batch, latent]
                z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
            elif ingau == 2:
                _mu, _logvar = self.directive_prior_net(x)  # [batch, latent]
                mu, logvar = self.directive_recognize_net(x, y)  # [batch, latent]
                z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
            elif ingau == 3:
                _mu, _logvar = self.commissive_prior_net(x)  # [batch, latent]
                mu, logvar = self.commissive_recognize_net(x, y)  # [batch, latent]
                z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
            else:
                _mu, _logvar = self.prior_net(x)  # [batch, latent]
                mu, logvar = self.recognize_net(x, y)  # [batch, latent]
                z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]


            # 解码器的输入为回复去掉end_id
            decoder_inputs = embed_responses[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            decoder_inputs = decoder_inputs.split([1] * len_decoder, 0)  # 解码器每一步的输入 seq-1个[1, batch, embed_size]
            outputs = []
            for idx in range(len_decoder):
                if idx == 0:
                    if ingau == 0:
                        state = self.inform_prepare_state(torch.cat([z, x], 1))  # 解码器初始状态
                    elif ingau == 1:
                        state = self.question_prepare_state(torch.cat([z, x], 1))  # 解码器初始状态
                    elif ingau == 2:
                        state = self.directive_prepare_state(torch.cat([z, x], 1))  # 解码器初始状态
                    elif ingau == 4:
                        state = self.prepare_state(torch.cat([z, x], 1))  # 解码器初始状态
                    else:
                        state = self.commissive_prepare_state(torch.cat([z, x], 1))  # 解码器初始状态
                decoder_input = decoder_inputs[idx]  # 当前时间步输入 [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                if ingau == 0:
                    output, state = self.inform_decoder(decoder_input, state)
                elif ingau == 1:
                    output, state = self.question_decoder(decoder_input, state)
                elif ingau == 2:
                    output, state = self.directive_decoder(decoder_input, state)
                elif ingau == 4:
                    output, state = self.decoder(decoder_input, state)
                else:
                    output, state = self.commissive_decoder(decoder_input, state)
                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]
            if ingau == 0:
                output_vocab = self.projector_inform(outputs)  # [batch, seq-1, num_vocab]
            elif ingau == 1:
                output_vocab = self.projector_question(outputs)  # [batch, seq-1, num_vocab]
            elif ingau == 2:
                output_vocab = self.projector_directive(outputs)  # [batch, seq-1, num_vocab]
            elif ingau == 4:
                output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]
            else:
                output_vocab = self.projector_commssive(outputs)  # [batch, seq-1, num_vocab]



            return output_vocab, _mu, _logvar, mu, logvar, classify_result
        else:  # 测试
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            id_catgory = torch.tensor([0,1,2,3]).long().repeat(sampled_latents.size(0),1).cuda()#[batch,cat_num]
            batch_size = id_posts.size(0)

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_catgory = word2vec.embedding(id_catgory) #[batch, cat_num, embed_size]
            # state = [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            if isinstance(state_posts, tuple):  # 如果是lstm则取h
                state_posts = state_posts[0]  # [layers, batch, dim]
            x = state_posts[-1, :, :]  # 取最后一层 [batch, dim]

            _mu, _logvar = self.prior_net(x)  # [batch, latent]
            z = _mu + (0.5 * _logvar).exp() * sampled_latents  # [batch, latent]

            inform_mu, inform_logvar = self.inform_prior_net(x)  # [batch, latent]
            inform_z = inform_mu + (0.5 * inform_logvar).exp() * sampled_latents  # [batch, latent]

            question_mu, question_logvar = self.question_prior_net(x)  # [batch, latent]
            question_z = question_mu + (0.5 * question_logvar).exp() * sampled_latents  # [batch, latent]

            directive_mu, directive_logvar = self.directive_prior_net(x)  # [batch, latent]
            directive_z = directive_mu + (0.5 * directive_logvar).exp() * sampled_latents  # [batch, latent]

            commissive_mu, commissive_logvar = self.commissive_prior_net(x)  # [batch, latent]
            commissive_z = commissive_mu + (0.5 * commissive_logvar).exp() * sampled_latents  # [batch, latent]

            ms_mu, ms_logvar = self.ms_prior_net(x)  # [batch, latent]
            ms_z = ms_mu + (0.5 * ms_logvar).exp() * sampled_latents  # [batch, latent]

            classify_result = self.classification(x)  # [batch, seq, num_catgory]
            clf_result = classify_result.argmax(1).detach().tolist()
            classify_result = torch.unsqueeze(classify_result, dim=2)
            input_cat = torch.bmm(embed_catgory.transpose(1, 2), classify_result).transpose(1, 2).transpose(0,1)  # [1, batch, embed_size]

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]
            done = torch.tensor([0] * batch_size).bool()
            first_input_id = (torch.ones((1, batch_size)) * self.config.start_id).long()
            if gpu:
                done = done.cuda()
                first_input_id = first_input_id.cuda()

            outputs_0, outputs_1, outputs_2, outputs_3, outputs, outputs_ms = [], [], [], [], [], []
            for num_catgory in range(0, act_num):
                done = torch.tensor([0] * batch_size).bool().cuda()
                if num_catgory == 0:
                    for idx in range(max_len):
                        if idx == 0:  # 第一个时间步
                            state = self.inform_prepare_state(torch.cat([inform_z, x], 1))  # 解码器初始状态
                            decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                        else:
                            decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                        # output: [1, batch, dim_out]
                        # state: [num_layers, batch, dim_out]
                        output, state = self.inform_decoder(decoder_input, state)
                        outputs_0.append(output)

                        vocab_prob = self.projector_inform(output)  # [1, batch, num_vocab]
                        next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                        _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                        done = done | _done  # 所有完成解码的
                        if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                            break

                    outputs_0 = torch.cat(outputs_0, 0).transpose(0, 1)  # [batch, seq, dim_out]
                    outputs_0 = self.projector_inform(outputs_0)  # [batch, seq, num_vocab]
                elif num_catgory == 1:
                    for idx in range(max_len):
                        if idx == 0:  # 第一个时间步
                            state = self.question_prepare_state(torch.cat([question_z, x], 1))  # 解码器初始状态
                            decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                        else:
                            decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                        # output: [1, batch, dim_out]
                        # state: [num_layers, batch, dim_out]
                        output, state = self.question_decoder(decoder_input, state)
                        outputs_1.append(output)

                        vocab_prob = self.projector_question(output)  # [1, batch, num_vocab]
                        next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                        _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                        done = done | _done  # 所有完成解码的
                        if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                            break

                    outputs_1 = torch.cat(outputs_1, 0).transpose(0, 1)  # [batch, seq, dim_out]
                    outputs_1 = self.projector_question(outputs_1)  # [batch, seq, num_vocab]
                elif num_catgory == 2:
                    for idx in range(max_len):
                        if idx == 0:  # 第一个时间步
                            state = self.directive_prepare_state(torch.cat([directive_z, x], 1))  # 解码器初始状态
                            decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                        else:
                            decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                        # output: [1, batch, dim_out]
                        # state: [num_layers, batch, dim_out]
                        output, state = self.directive_decoder(decoder_input, state)
                        outputs_2.append(output)

                        vocab_prob = self.projector_directive(output)  # [1, batch, num_vocab]
                        next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                        _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                        done = done | _done  # 所有完成解码的
                        if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                            break

                    outputs_2 = torch.cat(outputs_2, 0).transpose(0, 1)  # [batch, seq, dim_out]
                    outputs_2 = self.projector_directive(outputs_2)  # [batch, seq, num_vocab]
                else:
                    for idx in range(max_len):
                        if idx == 0:  # 第一个时间步
                            state = self.commissive_prepare_state(torch.cat([commissive_z, x], 1))  # 解码器初始状态
                            decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                        else:
                            decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                        # output: [1, batch, dim_out]
                        # state: [num_layers, batch, dim_out]
                        output, state = self.commissive_decoder(decoder_input, state)
                        outputs_3.append(output)

                        vocab_prob = self.projector_commssive(output)  # [1, batch, num_vocab]
                        next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                        _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                        done = done | _done  # 所有完成解码的
                        if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                            break

                    outputs_3 = torch.cat(outputs_3, 0).transpose(0, 1)  # [batch, seq, dim_out]
                    outputs_3 = self.projector_commssive(outputs_3)  # [batch, seq, num_vocab]
            done = torch.tensor([0] * batch_size).bool().cuda()
            for idx in range(max_len):
                if idx == 0:  # 第一个时间步
                    state = first_state  # 解码器初始状态
                    decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                else:
                    decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layers, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq, dim_out]
            outputs = self.projector(outputs)  # [batch, seq, num_vocab]

            done = torch.tensor([0] * batch_size).bool().cuda()
            for idx in range(max_len):
                if idx == 0:  # 第一个时间步
                    state = self.ms_prepare_state(torch.cat([ms_z, x], 1))  # 解码器初始状态
                    decoder_input = word2vec.embedding(first_input_id)  # 解码器初始输入 [1, batch, embed_size]
                    decoder_input = self.input_cat_pre(torch.cat((decoder_input, input_cat), 2))
                else:
                    decoder_input = word2vec.embedding(next_input_id)  # [1, batch, embed_size]
                # output: [1, batch, dim_out]
                # state: [num_layers, batch, dim_out]
                output, state = self.decoder(decoder_input, state)
                output = self.input_cat_pre(torch.cat((output, input_cat), 2))
                outputs_ms.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs_ms = torch.cat(outputs_ms, 0).transpose(0, 1)  # [batch, seq, dim_out]
            outputs_ms = self.projector(outputs_ms)  # [batch, seq, num_vocab]

            return outputs_0, outputs_1, outputs_2, outputs_3, outputs, outputs_ms , clf_result

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
                    'response_encoder': self.response_encoder.state_dict(),
                    'classification': self.classification.state_dict(),
                    #隐空间
                    'prior_net': self.prior_net.state_dict(),
                    'recognize_net': self.recognize_net.state_dict(),
                    'inform_prior_net': self.inform_prior_net.state_dict(),
                    'inform_recognize_net': self.inform_recognize_net.state_dict(),
                    'question_prior_net': self.question_prior_net.state_dict(),
                    'question_recognize_net': self.question_recognize_net.state_dict(),
                    'directive_prior_net': self.directive_prior_net.state_dict(),
                    'directive_recognize_net': self.directive_recognize_net.state_dict(),
                    'commissive_prior_net': self.commissive_prior_net.state_dict(),
                    'commissive_recognize_net': self.commissive_recognize_net.state_dict(),
                    'ms_prior_net': self.ms_prior_net.state_dict(),
                    'ms_recognize_net': self.ms_recognize_net.state_dict(),
                    #初始化隐态
                    'prepare_state': self.prepare_state.state_dict(),
                    'inform_prepare_state': self.inform_prepare_state.state_dict(),
                    'question_prepare_state': self.question_prepare_state.state_dict(),
                    'directive_prepare_state': self.directive_prepare_state.state_dict(),
                    'commissive_prepare_state': self.commissive_prepare_state.state_dict(),
                    'ms_prepare_state': self.ms_prepare_state.state_dict(),
                    #ms输入
                    'input_cat_pre': self.input_cat_pre.state_dict(),
                    #解码器
                    'decoder': self.decoder.state_dict(),
                    'inform_decoder': self.inform_decoder.state_dict(),
                    'question_decoder': self.question_decoder.state_dict(),
                    'directive_decoder': self.directive_decoder.state_dict(),
                    'commissive_decoder': self.commissive_decoder.state_dict(),
                    'ms_decoder': self.ms_decoder.state_dict(),
                    #输出映射
                    'projector': self.projector.state_dict(),
                    'projector_inform': self.projector_inform.state_dict(),
                    'projector_question': self.projector_question.state_dict(),
                    'projector_commssive': self.projector_commssive.state_dict(),
                    'projector_directive': self.projector_directive.state_dict(),
                    'projector_ms': self.projector_ms.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.post_encoder.load_state_dict(checkpoint['post_encoder'])
        self.response_encoder.load_state_dict(checkpoint['response_encoder'])
        self.classification.load_state_dict(checkpoint['classification'])
        #隐空间
        self.prior_net.load_state_dict(checkpoint['prior_net'])
        self.recognize_net.load_state_dict(checkpoint['recognize_net'])
        self.inform_prior_net.load_state_dict(checkpoint['inform_prior_net'])
        self.inform_recognize_net.load_state_dict(checkpoint['inform_recognize_net'])
        self.question_prior_net.load_state_dict(checkpoint['question_prior_net'])
        self.question_recognize_net.load_state_dict(checkpoint['question_recognize_net'])
        self.directive_prior_net.load_state_dict(checkpoint['directive_prior_net'])
        self.directive_recognize_net.load_state_dict(checkpoint['directive_recognize_net'])
        self.commissive_prior_net.load_state_dict(checkpoint['commissive_prior_net'])
        self.commissive_recognize_net.load_state_dict(checkpoint['commissive_recognize_net'])
        self.ms_prior_net.load_state_dict(checkpoint['ms_prior_net'])
        self.ms_recognize_net.load_state_dict(checkpoint['ms_recognize_net'])
        #初始化解码器初始隐态
        self.prepare_state.load_state_dict(checkpoint['prepare_state'])
        self.inform_prepare_state.load_state_dict(checkpoint['inform_prepare_state'])
        self.question_prepare_state.load_state_dict(checkpoint['question_prepare_state'])
        self.directive_prepare_state.load_state_dict(checkpoint['directive_prepare_state'])
        self.commissive_prepare_state.load_state_dict(checkpoint['commissive_prepare_state'])
        self.ms_prepare_state.load_state_dict(checkpoint['ms_prepare_state'])
        # ms输入
        self.input_cat_pre.load_state_dict(checkpoint['input_cat_pre'])
        #解码器
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.inform_decoder.load_state_dict(checkpoint['inform_decoder'])
        self.question_decoder.load_state_dict(checkpoint['question_decoder'])
        self.directive_decoder.load_state_dict(checkpoint['directive_decoder'])
        self.commissive_decoder.load_state_dict(checkpoint['commissive_decoder'])
        self.ms_decoder.load_state_dict(checkpoint['ms_decoder'])
        #输出映射
        self.projector.load_state_dict(checkpoint['projector'])
        self.projector_inform.load_state_dict(checkpoint['projector_inform'])
        self.projector_question.load_state_dict(checkpoint['projector_question'])
        self.projector_directive.load_state_dict(checkpoint['projector_directive'])
        self.projector_commssive.load_state_dict(checkpoint['projector_commssive'])
        self.projector_ms.load_state_dict(checkpoint['projector_ms'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        return epoch, global_step

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.config.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)




    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

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

def cluster_acc(Y_pred, Y):
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    from .util.linear_assignment_ import linear_assignment
    # 添加as语句不用修改代码中的函数名
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

