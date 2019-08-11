import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)  # [1, 13], 最大值
    return idx.item()  # 最大值的索引


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]  # 最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # [1, 13]  13个最大值组成的tensor
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))  # [13, 13]

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 第12行等于 -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 第13列等于 -10000
        # print(self.transitions.data)

        self.hidden = self.init_hidden()  # 初始化hidden ([2, 1, 100], [2, 1, 100])

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # 使用前向算法来计算分区函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)  # [1, 13], 内容-10000.
        # START_TAG拥有所有分数
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.  # init_alphas[0][11]

        # 包装一个变量，以便我们获得自动反向提升
        forward_var = init_alphas
        # forward_var: tensor([[-10000., -10000., -10000., -10000., -10000., -10000., -10000., -10000.,
        #                       -10000., -10000., -10000.,      0., -10000.]])

        # 句子迭代
        for feat in feats:  # 60  按行迭代
            alphas_t = []  # 在这个时间步的前向张量
            for next_tag in range(self.tagset_size):
                # 广播发射得分：无论以前的标记如何都是相同的
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # [-0.0879] -> [[-0.0879]] -> (1,13)
                # trans_score的第i个条目是从i转换到next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)  # [13] -> [1, 13]
                # next_tag_var的第i个条目是我们执行log-sum-exp之前的边（i  - > next_tag）的值
                next_tag_var = forward_var + trans_score + emit_score  # [1, 13]
                # 此标记的转发变量是所有分数的log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)  # [1, 13]
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # [1, 13]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()  # ([2, 1, 100], [2, 1, 100])
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # [60, 100] -> [60, 1, 100]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # [60, 1, 200]  ([2, 1, 100], [2, 1, 100])
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # [60, 200]
        lstm_feats = self.hidden2tag(lstm_out)  # [60, 13]
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)  # tensor([0.])
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # 11, [60] -> [61]  # 把11加到tags中
        # tensor([11,  8,  7,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0, ..., 0])
        for i, feat in enumerate(feats):  # [60, 13]
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            # tensor([0.1078], grad_fn=<AddBackward0>)
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # 初始化
        init_vvars = torch.full((1, self.tagset_size), -10000.)  # [1, 13]
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        # tensor([[-10000.,... , -10000.,      0., -10000.]])

        # 步骤i中的forward_var保持步骤i-1的维特比变量
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i]保存上一步中tag i的viterbi变量，加上从tag i转换到next_tag的分数。
                # 我们这里不包括排放分数，因为最大值不依赖于它们
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 现在添加排放分数，并将forward_var分配给我们刚刚计算的维特比变量集
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # [60], [60]
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)  # tensor(165.4411, grad_fn=<AddBackward0>)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
