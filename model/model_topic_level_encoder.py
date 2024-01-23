import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np
from model.Embedding import Embedding
from model.Encoder import Encoder
from model.PriorNetTopic import PriorNet
from model.RecognizeNetTopic import RecognizeNet
from model.Decoder import Decoder
from model.PrepareState import PrepareState
import torch.nn.functional as F

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden, encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

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

        #topic编码器
        self.topic_encoder = Encoder(config.response_encoder_cell_type,
                                        config.embedding_size*2,  # 输入维度
                                        config.response_encoder_output_size,  # 输出维度
                                        config.response_encoder_num_layers,  # rnn层数
                                        config.response_encoder_bidirectional,  # 是否双向
                                        config.dropout)  # dropout概率

        # word编码器
        self.word_encoder = Encoder(config.response_encoder_cell_type,
                                     config.embedding_size,  # 输入维度
                                     config.response_encoder_output_size,  # 输出维度
                                     config.response_encoder_num_layers,  # rnn层数
                                     config.response_encoder_bidirectional,  # 是否双向
                                     config.dropout)  # dropout概率

        # sentence编码器
        self.sentence_encoder = Encoder(config.response_encoder_cell_type,
                                     config.embedding_size,  # 输入维度
                                     config.response_encoder_output_size,  # 输出维度
                                     config.response_encoder_num_layers,  # rnn层数
                                     config.response_encoder_bidirectional,  # 是否双向
                                     config.dropout)  # dropout概率

        #topic attention
        self.linear_one = Attn('concat', config.embedding_size)
        self.linear_two = Attn('concat', config.embedding_size)
        self.linear_three = Attn('concat', config.embedding_size)


        # 先验网络
        self.prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度300
                                          config.response_encoder_output_size,  # response输入维度300
                                          config.latent_size,  # 潜变量维度200
                                          config.dims_recognize)  # 隐藏层维度

        # 识别网络
        self.recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度300
                                          config.response_encoder_output_size,  # response输入维度300
                                          config.response_encoder_output_size,
                                          config.latent_size,  # 潜变量维度200
                                          config.dims_recognize)  # 隐藏层维度250

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

        # 高斯混合模型
        self.pi_ = nn.Parameter(torch.FloatTensor(config.nClusters, ).fill_(1) / config.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(config.nClusters, config.latent_size).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(config.nClusters, config.latent_size).fill_(0),
                                         requires_grad=True)

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax(-1)
        )

        #当前vae的u和sigma
        self.mu1 = None
        self.sigma1 = None

    def forward(self, inputs, word2vec, level_encoder_data, inference=False, inpre=False, ingau=False, max_len=60, gpu=True):
        if not inference:  # 训练
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            id_responses = inputs['responses']  # [batch, seq]
            len_responses = inputs['len_responses']  # [batch, seq]
            id_keywords = inputs['keywords']  # [batch, seq]
            len_keywords = inputs['len_keywords']  # [batch, seq]
            id_topic = inputs['topic']  # [batch, 1]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]

            len_decoder = id_responses.size(1) - 1

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_responses = word2vec.embedding(id_responses)  # [batch, seq, embed_size]
            embed_keywords = word2vec.embedding(id_keywords)  # [batch, seq, embed_size]
            embed_topic = word2vec.embedding(id_topic)  # [batch, 1, embed_size]

            #level encoder data


            level_sentences = level_encoder_data['id_sentences']  #2000, seq
            level_len_sentences = level_encoder_data['len_sentences']# 2000

            embed_level_sentences = word2vec.embedding(level_sentences)#2000, seq, emb

            #词级编码器
            _, state_word_level = self.word_encoder(embed_level_sentences.transpose(0, 1), level_len_sentences)
            if isinstance(state_word_level, tuple):
                state_word_level = state_word_level[0]
            sentence_encoder_input = state_word_level[-1, :, :]  # [2000, emb]
            sentence_encoder_input = sentence_encoder_input.unsqueeze(dim=0) #[batch 2000 emb]

            #句子级编码器
            len_sentence = torch.tensor([sentence_encoder_input.size()[1]]).long().cuda()
            _, state_sentence_level = self.word_encoder(sentence_encoder_input.transpose(0, 1),len_sentence)
            if isinstance(state_sentence_level, tuple):
                state_sentence_level = state_sentence_level[0]
            global_sem = state_sentence_level[-1, :, :]  # [batch, emb]
            global_sem = global_sem.unsqueeze(dim=0)

            # keyword attention  [batch,seq,embed_size]
            a_i_j_one = []
            for i in range(0, id_keywords.size()[1]):
                a_i_j_one.append(
                    self.linear_one(embed_keywords[:, i, :].unsqueeze(dim=1).repeat(1, id_keywords.size()[1], 1),
                                    embed_keywords)) # seq 1 batch
            a_i_j_one = torch.stack(a_i_j_one,dim=1).squeeze(dim=2).transpose(0, 2) #batch seq seq
            ri = a_i_j_one.bmm(embed_keywords) # [batch, seq, embed_size]

            a_i_j_two = []
            for i in range(0, embed_topic.size()[1]):
                a_i_j_two.append(
                    self.linear_two(embed_topic[:, i, :].unsqueeze(dim=1).repeat(1, ri.size()[1], 1),
                                    ri)) # seq 1 batch

            a_i_j_two = torch.stack(a_i_j_two, dim=1).squeeze(dim=2).transpose(0, 2).transpose(1,2)  # batch 1 seq
            ti = a_i_j_two.bmm(embed_topic)  # [batch, seq, embed_size]

            a_i_j_three = []
            for i in range(0, global_sem.size()[1]):
                a_i_j_three.append(
                    self.linear_three(global_sem[:, i, :].unsqueeze(dim=1).repeat(1, ti.size()[1], 1),
                                    ti))  # seq 1 batch

            a_i_j_three = torch.stack(a_i_j_three, dim=1).squeeze(dim=2).transpose(0, 2).transpose(1, 2)  # batch 1 seq
            si = a_i_j_three.bmm(global_sem)  # [batch, seq, embed_size]

            ui = torch.cat((si, embed_topic.repeat(1, ti.size()[1],1)), dim=2)



            # keyword attention with topic attention
            # topic encoder
            # state: [layers, batch, dim]
            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
            _, state_keywords = self.topic_encoder(ui.transpose(0, 1), len_keywords)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_responses, tuple):
                state_responses = state_responses[0]
            if isinstance(state_keywords, tuple):
                state_keywords = state_keywords[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            y = state_responses[-1, :, :]  # [batch, dim]
            t = state_keywords[-1, :, :]

            # p(z|x)
            _mu, _logvar = self.prior_net(x, t)  # [batch, latent]

            # p(z|x,y)
            mu, logvar = self.recognize_net(x, y, t)  # [batch, latent]
            self.mu1 = mu
            self.sigma1 = logvar
            # 重参数化
            z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]

            pi = self.pi_
            log_sigma2_c = self.log_sigma2_c
            mu_c = self.mu_c
            det = 1e-10

            #z = torch.randn_like(mu) * torch.exp(logvar / 2) + mu
            yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

            yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

            Loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                                 torch.exp(logvar.unsqueeze(
                                                                     1) - log_sigma2_c.unsqueeze(0)) +
                                                                 (mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(
                                                                     2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))
            # 第二项
            Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
                torch.sum(1 + logvar, 1))
            # torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))第三个
            # 0.5*torch.mean(torch.sum(1+z_sigma2_log,1))第四个

            #print(np.argmax(yita_c.detach().cpu().numpy(),axis=1))

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


            return output_vocab, _mu, _logvar, mu, logvar, Loss
        else:  # 测试
            id_posts = inputs['posts']  # [batch, seq]
            len_posts = inputs['len_posts']  # [batch]
            sampled_latents = inputs['sampled_latents']  # [batch, latent_size]
            id_keywords = inputs['keywords']  # [batch, seq]
            len_keywords = inputs['len_keywords']  # [batch, seq]
            id_topic = inputs['topic']  # [batch, 1]
            batch_size = id_posts.size(0)

            embed_posts = word2vec.embedding(id_posts)  # [batch, seq, embed_size]
            embed_keywords = word2vec.embedding(id_keywords)  # [batch, seq, embed_size]
            embed_topic = word2vec.embedding(id_topic)  # [batch, 1, embed_size]

            # level encoder data

            level_sentences = level_encoder_data['id_sentences']  # 2000, seq
            level_len_sentences = level_encoder_data['len_sentences']  # 2000

            embed_level_sentences = word2vec.embedding(level_sentences)  # 2000, seq, emb

            # 词级编码器
            _, state_word_level = self.word_encoder(embed_level_sentences.transpose(0, 1), level_len_sentences)
            if isinstance(state_word_level, tuple):
                state_word_level = state_word_level[0]
            sentence_encoder_input = state_word_level[-1, :, :]  # [2000, emb]
            sentence_encoder_input = sentence_encoder_input.unsqueeze(dim=0)  # [batch 2000 emb]

            # 句子级编码器
            len_sentence = torch.tensor([sentence_encoder_input.size()[1]]).long().cuda()
            _, state_sentence_level = self.word_encoder(sentence_encoder_input.transpose(0, 1), len_sentence)
            if isinstance(state_sentence_level, tuple):
                state_sentence_level = state_sentence_level[0]
            global_sem = state_sentence_level[-1, :, :]  # [batch, emb]
            global_sem = global_sem.unsqueeze(dim=0)

            # keyword attention  [batch,seq,embed_size]
            a_i_j_one = []
            for i in range(0, id_keywords.size()[1]):
                a_i_j_one.append(
                    self.linear_one(embed_keywords[:, i, :].unsqueeze(dim=1).repeat(1, id_keywords.size()[1], 1),
                                    embed_keywords))  # seq 1 batch
            a_i_j_one = torch.stack(a_i_j_one, dim=1).squeeze(dim=2).transpose(0, 2)  # batch seq seq
            ri = a_i_j_one.bmm(embed_keywords)  # [batch, seq, embed_size]

            a_i_j_two = []
            for i in range(0, embed_topic.size()[1]):
                a_i_j_two.append(
                    self.linear_one(embed_topic[:, i, :].unsqueeze(dim=1).repeat(1, ri.size()[1], 1),
                                    ri))  # seq 1 batch

            a_i_j_two = torch.stack(a_i_j_two, dim=1).squeeze(dim=2).transpose(0, 2).transpose(1, 2)  # batch 1 seq
            ti = a_i_j_two.bmm(embed_topic)  # [batch, seq, embed_size]

            a_i_j_three = []
            for i in range(0, global_sem.size()[1]):
                a_i_j_three.append(
                    self.linear_three(global_sem[:, i, :].unsqueeze(dim=1).repeat(1, ti.size()[1], 1),
                                      ti))  # seq 1 batch

            a_i_j_three = torch.stack(a_i_j_three, dim=1).squeeze(dim=2).transpose(0, 2).transpose(1, 2)  # batch 1 seq
            si = a_i_j_three.bmm(global_sem)  # [batch, seq, embed_size]

            ui = torch.cat((si, embed_topic.repeat(1, ti.size()[1], 1)), dim=2)

            # state = [layers, batch, dim]

            _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
            _, state_keywords = self.topic_encoder(ui.transpose(0, 1), len_keywords)
            if isinstance(state_posts, tuple):
                state_posts = state_posts[0]
            if isinstance(state_keywords, tuple):
                state_keywords = state_keywords[0]
            x = state_posts[-1, :, :]  # [batch, dim]
            t = state_keywords[-1, :, :]

            # p(z|x)
            _mu, _logvar = self.prior_net(x, t)  # [batch, latent]
            # 重参数化
            z = _mu + (0.5 * _logvar).exp() * sampled_latents  # [batch, latent]

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]
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
                output, state = self.decoder(decoder_input, state)
                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # 选择概率最大的词作为下个时间步的输入 [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # 当前时间步完成解码的 [batch]
                done = done | _done  # 所有完成解码的
                if done.sum() == batch_size:  # 如果全部解码完成则提前停止
                    break

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq, dim_out]
            output_vocab = self.projector(outputs)  # [batch, seq, num_vocab]

            return output_vocab, _mu, _logvar, None, None

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

    def predict(self,inputs):
        id_posts = inputs['posts']  # [batch, seq]
        len_posts = inputs['len_posts']  # [batch]
        id_responses = inputs['responses']  # [batch, seq]
        len_responses = inputs['len_responses']  # [batch, seq]
        sampled_latents = inputs['sampled_latents']  # [batch, latent_size]

        embed_posts = self.embedding(id_posts)  # [batch, seq, embed_size]
        embed_responses = self.embedding(id_responses)  # [batch, seq, embed_size]
        # state: [layers, batch, dim]
        _, state_posts = self.post_encoder(embed_posts.transpose(0, 1), len_posts)
        _, state_responses = self.response_encoder(embed_responses.transpose(0, 1), len_responses)
        if isinstance(state_posts, tuple):
            state_posts = state_posts[0]
        if isinstance(state_responses, tuple):
            state_responses = state_responses[0]
        x = state_posts[-1, :, :]  # [batch, dim]
        y = state_responses[-1, :, :]  # [batch, dim]

        # p(z|x,y)
        z_mu, z_sigma2_log = self.recognize_net(x, y)  # [batch, latent]
        # 重参数化
        z = z_mu + (0.5 * z_sigma2_log).exp() * sampled_latents  # [batch, latent]
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)

    def save_model(self, epoch, global_step, path):
        r""" 保存模型 """
        torch.save({'embedding': self.embedding.state_dict(),
                    'post_encoder': self.post_encoder.state_dict(),
                    'response_encoder': self.response_encoder.state_dict(),
                    'prior_net': self.prior_net.state_dict(),
                    'recognize_net': self.recognize_net.state_dict(),
                    'prepare_state': self.prepare_state.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'projector': self.projector.state_dict(),
                    'topic_encoder': self.topic_encoder.state_dict(),
                    'word_encoder': self.word_encoder.state_dict(),
                    'sentence_encoder': self.sentence_encoder.state_dict(),
                    'linear_one': self.linear_one.state_dict(),
                    'linear_teo': self.linear_two.state_dict(),
                    'linear_three':self.linear_three.state_dict(),
                    'pi_': self.pi_,
                    'mu_c': self.mu_c,
                    'log_sigma2_c': self.log_sigma2_c,
                    'mu1': self.mu1,
                    'sigma1': self.sigma1,
                    'epoch': epoch,
                    'global_step': global_step}, path)

    def load_model(self, path):
        r""" 载入模型 """
        checkpoint = torch.load(path)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.post_encoder.load_state_dict(checkpoint['post_encoder'])
        self.response_encoder.load_state_dict(checkpoint['response_encoder'])
        self.prior_net.load_state_dict(checkpoint['prior_net'])
        self.recognize_net.load_state_dict(checkpoint['recognize_net'])
        self.prepare_state.load_state_dict(checkpoint['prepare_state'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        self.topic_encoder.load_state_dict(checkpoint['topic_encoder'])
        self.word_encoder.load_state_dict(checkpoint['word_encoder'])
        self.sentence_encoder.load_state_dict(checkpoint['sentence_encoder'])
        self.linear_three.load_state_dict(checkpoint['linear_three'])
        self.linear_one.load_state_dict(checkpoint['linear_one'])
        self.linear_two.load_state_dict(checkpoint['linear_teo'])
        self.pi_ = checkpoint['pi_']
        self.mu_c = checkpoint['mu_c']
        self.log_sigma2_c = checkpoint['log_sigma2_c']
        self.mu1 = checkpoint['mu1']
        self.sigma1 = checkpoint['sigma1']
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

