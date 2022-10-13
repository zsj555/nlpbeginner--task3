
import torch
import torch.nn as nn


class Input_Encoding(nn.Module):
    def __init__(self, embedding_dim, len_hidden, len_words, longest, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.longest=longest
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, embedding_dim))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=embedding_dim, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=embedding_dim, _weight=weight).cuda()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()

    def forward(self, x):
        x = torch.LongTensor(x.cpu().numpy()).cuda()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class LocalInference(nn.Module):
    def __init__(self):
        """
        p代表：premise
        h代表：hypothesis
        """

        super(LocalInference, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1).cuda()
        self.softmax_2 = nn.Softmax(dim=2).cuda()



    def forward(self, p, h):
        e = torch.matmul(p, h.transpose(1, 2)).cuda()

        p_ = self.softmax_2(e)
        p_ = p_.bmm(h)
        h_ = self.softmax_1(e)
        h_ = h_.transpose(1, 2).bmm(p)

        maskp = torch.cat([p, p_, p - p_, p * p_], dim=-1)
        maskh = torch.cat([h, h_, h - h_, h * h_], dim=-1)

        return maskp, maskh

class Inference_Composition(nn.Module):
    def __init__(self, embedding_dim, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):

        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(len_hidden_m, embedding_dim).cuda()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()
        self.dropout = nn.Dropout(drop_out).cuda()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)

        return output

class Prediction(nn.Module):
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(),
                                 nn.Linear(len_mid, type_num)).cuda()

    def forward(self, p, h):

        vp_avg = p.sum(1) / p.shape[1] #平均池
        vp_max = p.max(1)[0]       #最大池

        vh_avg = h.sum(1) / h.shape[1]
        vh_max = h.max(1)[0]

        out_put = torch.cat((vp_avg, vp_max, vh_avg, vh_max), dim=-1)

        return self.mlp(out_put)

class ESIM(nn.Module):
    def __init__(self, embedding_dim, len_hidden, len_words, longest, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words = len_words
        self.longest = longest
        self.input_encoding = Input_Encoding(embedding_dim, len_hidden, len_words, longest, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        self.localInference = LocalInference()
        self.inference_composition = Inference_Composition(embedding_dim, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        self.prediction = Prediction(len_hidden*8, len_hidden, type_num=type_num, drop_out=drop_out)

    def forward(self,p,h):
        p_bar = self.input_encoding(p)
        h_bar = self.input_encoding(h)

        maskp, maskh = self.localInference(p_bar, h_bar)

        v_p = self.inference_composition(maskp)
        v_h = self.inference_composition(maskh)

        out_put = self.prediction(v_p,v_h)

        return out_put