from torch import nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False,
                 lstm_dropout=0.0,
                 input_dropout=0.2
                 ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=input_dropout)

    def forward(self, x, x_len):
        # x should have descending seq_len
        x = self.dropout(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=x_len,
            batch_first=True,
            enforce_sorted=True
        )
        out_packed, (h_n, c_n) = self.lstm(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed,
            batch_first=True
        )
        return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, input_dropout=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(p=input_dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x


# TODO: finisth this part, no need gradient, always validation mode
class MTEmbedding(nn.Module):
    def __init__(self, encoder):
        super(MTEmbedding, self).__init__()
        self.encoder = encoder

    def forward(self, x, x_len):
        return self.encoder(x, x_len)[0]


class HignWay(nn.Module):
    def __init__(self, d):
        super(HignWay, self).__init__()
        self.linear_layers = []
        self.gate_layers = []
        for i in range(2):
            self.linear_layers.append(
                nn.Sequential(Linear(2 * d, 2 * d), nn.ReLU())
            )
            self.gate_layers.append(
                nn.Sequential(Linear(2 * d, 2 * d), nn.Sigmoid())
            )

    def forward(self, x):
        # x = torch.cat([x1, x2], dim=-1)
        for i in range(2):
            h = self.linear_layers[i](x)
            g = self.gate_layers[i](x)
            x = g * h + (1 - g) * x
        return x


class AttentionFlow(nn.Module):
    def __init__(self, d):
        super(AttentionFlow, self).__init__()
        self.weight_c = Linear(2 * d, 1)
        self.weight_q = Linear(2 * d, 1)
        self.weight_cq = Linear(2 * d, 1)

    def forward(self, c, q):
        c_len = c.size(1)
        q_len = q.size(1)

        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = self.weight_c(c).expand(-1, -1, q_len) + \
            self.weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x


class QAOutput(nn.Module):
    def __init__(self, d, dropout):
        super(QAOutput, self).__init__()
        self.linear_p1 = Linear(10 * d, 1, input_dropout=dropout)
        self.linear_p2 = Linear(10 * d, 1, input_dropout=dropout)
        self.output_lstm = LSTM(
            input_size=2 * d,
            hidden_size=d,
            bidirectional=True,
            input_dropout=dropout
        )

    def forward(self, g, m, l):
        gm = torch.cat([g, m], dim=-1)
        p1 = self.linear_p1(gm).squeeze()
        m2 = self.output_lstm(m, l)
        gm2 = torch.cat([g, m2], dim=-1)
        p2 = self.linear_p2(gm2).squeeze()
        return p1, p2


class BiDAF(nn.Module):
    def __init__(self,
                 word_emb,
                 cxt_emb=None,
                 hidden_dim=100,
                 dropout=0.2):
        super(BiDAF, self).__init__()

        # 1a. word Embedding layer
        self.word_emb = nn.Embedding.from_pretrained(word_emb, freeze=True)
        # 1b. cxt Embedding layer
        if cxt_emb is None:
            self.cxt_emb = None
        elif cxt_emb == 'mt_emb':
            # TODO
            pass
            # self.cxt_emb = CTXEmbedding()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # 2. Highway
        self.highway = HignWay(d=self.hidden_dim)

        # 3. Contextual Embedding Layer
        self.context_lstm = LSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            input_dropout=self.dropout)

        self.attention_flow = AttentionFlow(d=self.hidden_dim)

        self.modeling_lstm = LSTM(
            input_size=self.hidden_dim * 8,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            input_dropout=self.dropout
        )

        self.qa_output = QAOutput(
            d=self.hidden_dim,
            dropout=self.dropout
        )

    def forward(self, batch):
        context, context_len = batch.context[0], batch.context[1]
        question, question_len = batch.question[0], batch.question[1]

        # 1a. word Embedding Layer
        c = self.word_emb(context)
        q = self.word_emb(question)
        # 1b. cxt Embedding Layer
        if self.cxt_emb is not None:
            c_cxt = self.cxt_emb(context, context_len)
            q_cxt = self.cxt_emb(question, question_len)
            c = torch.cat([c, c_cxt], dim=-1)
            q = torch.cat([q, q_cxt], dim=-1)

        # 2. Highway network
        c = self.highway(c)
        q = self.highway(q)

        # 3. Contextual Embedding Layer
        c = self.context_LSTM(c, context_len)
        q = self.context_LSTM(q, question_len)

        # 4. Attention Flow Layer
        g = self.attention_flow(c, q)

        # 5. Modeling Layer
        m = self.modeling_LSTM(g, context_len)

        # 6. Output Layer
        p1, p2 = self.qa_output(g, m, context_len)

        # (batch, c_len), (batch, c_len)
        return p1, p2
