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

        self.input_cat_pre = nn.Linear(config.embedding_size*2, config.embedding_size)

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
        z_s = []
        label_s = []

        with torch.no_grad():
            for data in inputs.get_batch_data():
                inputs2 = self.prepare_feed_data(data)
                id_posts = inputs2['posts']  # [batch, seq]
                len_posts = inputs2['len_posts']  # [batch]
                id_responses = inputs2['responses']  # [batch, seq]
                len_responses = inputs2['len_responses']  # [batch, seq]
                sampled_latents = inputs2['sampled_latents']  # [batch, latent_size]
                id_responses_act = inputs2['post_act']
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
                ingau = id_responses_act.detach().cpu().numpy()
                if ingau[0][0] == 0:
                    _mu, _logvar = self.inform_prior_net(x)  # [batch, latent]
                    mu, logvar = self.inform_recognize_net(x, y)  # [batch, latent]
                    z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
                elif ingau[0][0] == 1:
                    _mu, _logvar = self.question_prior_net(x)  # [batch, latent]
                    mu, logvar = self.question_recognize_net(x, y)  # [batch, latent]
                    z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
                elif ingau[0][0] == 2:
                    _mu, _logvar = self.directive_prior_net(x)  # [batch, latent]
                    mu, logvar = self.directive_recognize_net(x, y)  # [batch, latent]
                    z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
                elif ingau[0][0] == 3:
                    _mu, _logvar = self.commissive_prior_net(x)  # [batch, latent]
                    mu, logvar = self.commissive_recognize_net(x, y)  # [batch, latent]
                    z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
                else:
                    _mu, _logvar = self.prior_net(x)  # [batch, latent]
                    mu, logvar = self.recognize_net(x, y)  # [batch, latent]
                    z = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]

                z_ = z.detach().cpu().numpy()
                z_s.append(z_)
                label_s.append(id_responses_act.detach().cpu().numpy())

        z_s = np.concatenate(z_s, 0)
        label_s = np.concatenate(label_s, 0)
        return z_s, label_s

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

    def prepare_feed_data(self, data, inference=False):
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
                         'post_act': torch.tensor(data['post_act']),
                         'post_emotion': torch.tensor(data['post_emotion'])}  # [batch, len_decoder]
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

