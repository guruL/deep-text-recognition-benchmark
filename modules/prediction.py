import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, feature_map, batch_H, hidden, text, is_train=True, batch_max_length=25):
        """
        input:
            feature_map : the last output of feature extractor. size [batch_size, channel, H, W]
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes] #???
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """

        batch_size = batch_H.size(0)
        num_steps = batch_max_length #+ 1 #for [s] at end of sentence. 

        output_hiddens = torch.FloatTensor(batch_size, num_steps+1, self.hidden_size).fill_(0).to(device)
        hidden_1 = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))
        hidden_2 = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))
                  # i add a LSTM, total 2 LTSM to decode embedding

        # 第一次提出来
        # 关键是，第一个 char_onehots
        # 这个地方怎么写更好，我也不知道。。。感觉好多可以考虑的方法
        char_onehots = self._char_to_onehot(text[:, 0], onehot_dim=self.num_classes)
        hidden_1 = ((hidden[0][0] + hidden[0][1]) / 2, (hidden[1][0] + hidden[1][1]) / 2)
        hidden_2 = ((hidden[0][0] + hidden[0][1]) / 2, (hidden[1][0] + hidden[1][1]) / 2)
        hidden_b = torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device)
        hidden_1, hidden_2, alpha = self.attention_cell(feature_map, hidden_1, hidden_2, batch_H, char_onehots)
        output_hiddens[:, 0, :] = hidden_2[0]

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i+1], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden_1, hidden_2, alpha = self.attention_cell(feature_map, hidden_1, hidden_2, batch_H, char_onehots)
                output_hiddens[:, i+1, :] = hidden_2[0]  # LSTM hidden index (0: hidden, 1: Cell)
                # hidden_2 is the final output of Attention
            probs = self.generator(output_hiddens)

        else:
            probs_step = self.generator(hidden_2[0])
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps+1, self.num_classes).fill_(0).to(device)
            probs[:, 0, :] = probs_step
            _, next_input = probs_step.max(1)
            targets = next_input

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden_1, hidden_2, alpha = self.attention_cell(feature_map, hidden_1, hidden_2, batch_H, char_onehots)
                probs_step = self.generator(hidden_2[0])
                probs[:, i+1, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn1 = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hlinear = nn.Linear(hidden_size, hidden_size)
        self.rnn2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size

        self.feature_map_channel = 512
        self.conv_h2h = nn.Conv2d(hidden_size, hidden_size, 1)
        self.conv_m2h = nn.Conv2d(self.feature_map_channel, hidden_size, 3, 1, 1)
        self.conv_s2a = nn.Conv2d(hidden_size, 1, 1)
        self.score = nn.Conv2d(hidden_size, 1, 1)

    def forward(self, feature_map, prev_hidden_1, prev_hidden_2, batch_H, char_onehots):
        # diff: add feature_map !!!
        # feature_map size [batch_size x channel x H x W] 
        # firstly, we should get batch_size, H and W from the feature map
        # we assume that the channel of feature are known, named self.feature_map_channel
        feature_batch_size, _, feature_map_H, feature_map_W = feature_map.size()
        feature_map_h = self.conv_m2h(feature_map)

        # prev_hidden  : (h, c)
        prev_hidden_proj = self.h2h(prev_hidden_2[0])#.unsqueeze(2).unsqueeze(2) # prev_hidden_2[0] shape [batch_size x hidden_size]

        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H) # batch_H shape [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = torch.mean(batch_H_proj, 1)
        batch_H_proj = batch_H_proj + prev_hidden_proj

        batch_H = batch_H_proj.repeat(feature_map_H, feature_map_W, 1, 1).permute(2, 3, 0, 1) # batch_size x channel x H x W

        batch_H_proj = self.conv_h2h(batch_H)
        

        
        # e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1
        e = self.score(torch.tanh(feature_map_h + batch_H_proj)) # + prev_hidden_proj))
        print(e.shape, feature_map_h.shape, batch_H_proj.shape, prev_hidden_proj.shape)
        e = e.reshape(feature_batch_size, 1, -1)
        alpha = F.softmax(e, dim=2)
        alpha = alpha.reshape(feature_batch_size, 1, feature_map_H, feature_map_W)

        context = torch.sum(torch.sum(torch.mul(alpha, feature_map_h), 3), 2)#.squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        
        cur_hidden_1 = self.rnn1(concat_context, prev_hidden_1)
        cur_hidden = self.hlinear(cur_hidden_1[0]) # LSTM hidden index (0: hidden, 1: Cell)
        cur_hidden_2 = self.rnn2(cur_hidden, prev_hidden_2)

        # from lstm decoder, we will get a hidden 

        return cur_hidden_1, cur_hidden_2, alpha
