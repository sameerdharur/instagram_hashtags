import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
import pickle
from torch.nn.utils.rnn import pack_padded_sequence

import config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens_c, embedding_tokens_h):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        with open(config.hashtags_vocabulary_path, 'rb') as fd:
            hash_vocab_json = pickle.load(fd)

        self.cnn = models.resnet50(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.cnn.layer4.register_forward_hook(save_output)

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens_c,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        print(vision_features)
        print(glimpses)
        print(question_features)
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=len(hash_vocab_json),
            drop=0.5,
        )
        self.hash = DecoderRNN(
                enc_hidden_size = glimpses * vision_features + question_features,
                dec_hidden_size = 256,
                output_size = len(hash_vocab_json)
            )
        # self.hash = HashtagProcessor(
        #     embedding_tokens=embedding_tokens,
        #     embedding_features=300,
        #     lstm_features=glimpses * vision_features + question_features,
        #     output_size = len(hash_vocab_json),
        #     drop=0.5,
        # )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, img, q, q_len, hashtag, a_len):
        self.cnn(img)
        v = self.buffer
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        # print(combined.shape)
        # ksljdfksd
        # answer = self.classifier(combined)
        predictions, hidden = self.hash(hashtag,combined)
        return predictions

class DecoderRNN(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        # self.enchidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, dec_hidden_size, padding_idx=0)
        self.gru = nn.GRU(dec_hidden_size*16*56, enc_hidden_size)
        self.out = nn.Linear(enc_hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(input)
        # input = input.to(torch.int64)
        # print(self.embedding.weight)
        print(input.shape)
        output = self.embedding(input)
        print(output.shape)
        output = output.view(1, 1, -1)
        output = F.relu(output)
        print(output.shape)
        print(hidden.shape)
        output, hidden = self.gru(output, hidden)
        prediction = self.out(output[0])
        return prediction, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))

class HashtagProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, output_size, drop=0.0):
        super(HashtagProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features
        self.out = nn.Linear(lstm_features, output_size)

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        output, hidden = self.lstm(packed)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        print(type(q))
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True, enforce_sorted = False)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s]
    weighted = attention * input # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
