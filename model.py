import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
import pickle
import random
from torch.nn.utils.rnn import pack_padded_sequence

import config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens_c, embedding_tokens_h, src_embed = None, trg_embed = None):
        super(Net, self).__init__()
        hidden_size = 512
        vision_features = config.output_features
        glimpses = 2
        dec_emb_size = 300
        question_features = dec_emb_size

        with open(config.hashtags_vocabulary_path, 'rb') as fd:
            self.hash_vocab_json = pickle.load(fd)


        self.output_size = len(self.hash_vocab_json)
        self.cnn = models.resnet50(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output

        self.cnn.layer4.register_forward_hook(save_output)
        self.linear = nn.Linear(self.cnn.fc.in_features, dec_emb_size)
        self.bn = nn.BatchNorm1d(dec_emb_size, momentum=0.01)
        #MAybe remove avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens_c,
            embedding_features=300,
            lstm_features=hidden_size,
            emb_weight_matrix = src_embed,
            drop=0.5
        )
        # print(vision_features)
        # print(glimpses)
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        # self.classifier = Classifier(
        #     in_features=glimpses * vision_features + question_features,
        #     mid_features=1024,
        #     out_features=self.output_size,
        #     drop=0.5,
        # )
        self.hash = DecoderRNN_IMGFeat(
                embed_size = dec_emb_size,
                hidden_size = hidden_size,
                output_size = self.output_size,
                emb_weight_matrix = trg_embed
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

    def forward(self, img, q, q_len, hashtag, a_len, teacher_forcing_ratio = 0.5):
        self.cnn(img)
        v = self.buffer
        v = self.avgpool(v)
        # print(a_len)
        # lkjsdgfd
        features = v.reshape(v.size(0), -1)
        features = self.bn(self.linear(features))
        q = q.permute(1,0)
        hashtag = hashtag.permute(1,0)
        h,c = self.text(q, list(q_len.data))
        # print(q.shape)
        # v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        # a = self.attention(v, q)
        # v = apply_attention(v, a)

        # combined = torch.cat([v, q], dim=1)
        # print(combined.shape)
        # ksljdfksd
        # answer = self.classifier(combined)
        #Don't pass combine as the input here
        #Use the actual features and caclulate attention based on those features
        # print(hashtag.shape)
        # if teacher_forcing_ratio == 0:
        #     hashtag = hashtag.permute(1,0)
        trg_len = hashtag.shape[0]
        batch_size = hashtag.shape[1]
        trg_vocab_size = self.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        input = hashtag[0,:]
        # print(input.shape)
        # if teacher_forcing_ratio == 0:
        #     input = hashtag[0,0]

        for t in range(1,trg_len):
            prediction, (h, c) = self.hash(features, input, h, c)
            # print(outputs.shape)
            # print(prediction.shape)
            outputs[t,:] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            # print(top1)
            # if teacher_forcing_ratio == 0:
            #     if top1 == self.hash_vocab_json['<eos>']:
            #         break
            # print(hashtag[t].shape)
            input = hashtag[t] if teacher_force else top1
        return outputs

class DecoderRNN(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        # self.enchidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, dec_hidden_size, padding_idx=0)
        self.gru = nn.GRU(dec_hidden_size, enc_hidden_size)
        self.out = nn.Linear(enc_hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(input)
        # input = input.to(torch.int64)
        # print(self.embedding.weight)
        # print(input.shape)
        output = self.embedding(input)
        # print(output.shape)
        # output = output.view(1, 1, -1)
        output = F.relu(output)
        # print(output.shape)
        # print(hidden.shape)
        output, hidden = self.gru(output, hidden)
        prediction = self.out(output[0])
        return prediction, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN_IMGFeat(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, emb_weight_matrix, num_layers= 1, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN_IMGFeat, self).__init__()
        self.embed = nn.Embedding(output_size, embed_size)
        if not len(emb_weight_matrix) == 0:
            self.embed.load_state_dict({'weight':torch.tensor(emb_weight_matrix)})
        self.lstm = nn.LSTM(2*embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, hashtags, h, c):
        """Decode image feature vectors and generates captions."""
        hashtags = hashtags.unsqueeze(0)
        embeddings = self.embed(hashtags)
        # print(features.unsqueeze(1).shape)
        # print(features.shape)
        # print(embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), 2)
        # print(embeddings.shape)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # print(h.shape)
        # print(c.shape)

        # print(h.permute(1,0,2)[0].shape)
        # sdsdf
        
        output, (h1,c1) = self.lstm(embeddings, (h,c))
        outputs = self.linear(output.squeeze(0))
        return outputs, (h1,c1)

    # def forward(self, features, captions, lengths, h,c):
    #     """Decode image feature vectors and generates captions."""
    #     embeddings = self.embed(captions)
    #     embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
    #     preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).cuda()
        
    #     for t in range(max_timespan):
    #         h, c = self.lstm(lstm_input, (h, c))
    #         output = self.linear(self.dropout(h))

    #     packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
    #     hiddens, _ = self.lstm(packed)
    #     outputs = self.linear(hiddens[0])
    #     return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids



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
    def __init__(self, embedding_tokens, embedding_features, lstm_features, emb_weight_matrix, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        if not len(emb_weight_matrix) == 0:
            self.embedding.load_state_dict({'weight':torch.tensor(emb_weight_matrix)})
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
        # print(type(q))
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        # packed = pack_padded_sequence(tanhed, q_len, batch_first=True, enforce_sorted = False)
        outputs, (h, c) = self.lstm(tanhed)
        return h,c


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
