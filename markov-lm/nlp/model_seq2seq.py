
# %%

# CHANGELOG
# v1: Adapted by @shouldsee into github.com/shouldsee/markov_lm nlp
# v0:code by Tae Hwan Jung @graykode
# source: https://github.com/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).py
# Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/14_2_seq2seq_att.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from markov_lm.util_html import write_png_tag
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


class Seq2SeqWithAttention(nn.Module):
    def __init__(self,device,config,_=None):
        super().__init__()
        self.config = config
        self.device = device
        self.n_hidden = n_hidden = config.embed_dim
        self.n_class  = n_class  = config.graph_dim
        dropout = config.beta
        assert config.depth == 1,config.depth
        self.embed = nn.Embedding(n_class,n_hidden).to(self.device)
        self.enc_cell = nn.RNN(num_layers=config.depth,input_size=n_hidden, hidden_size=n_hidden, dropout=dropout).to(self.device)
        self.dec_cell = nn.RNN(num_layers=config.depth,input_size=n_hidden, hidden_size=n_hidden, dropout=dropout).to(self.device)


        # Linear for attention
        self.attn = nn.Linear(n_hidden, n_hidden).to(self.device)
        self.out_layer  = nn.Linear(n_hidden * 2, n_class).to(self.device)
    def loss(self,item,):
        return self._loss(item,'loss')

    def forward(self,item):
        return self._loss(item,'forward')

    def _loss(self,item,ret):
        source = item['source'] ### token sequence
        target = item['target'] ### token seq
        # source_hot
        dec_input  = target[:,:-1]
        hidden = torch.zeros((1, len(source), self.n_hidden),device=self.device)


        output_logit, att_weight =self._forward(source, hidden, dec_input)
        if ret =='forward':
            return output_logit,att_weight
        # output_logp = self.out
        output_tok = item['target'][:,1:]
        loss = -torch.gather( output_logit.log_softmax(-1),index=output_tok.unsqueeze(-1),dim=-1).squeeze(-1)
        loss = loss.mean(-1)
        # import pdb; pdb.set_trace()
        return loss

    grad_loss = loss

    def _forward(self, enc_inputs, hidden, dec_inputs):
        # enc_inputs: [batch_size, n_step,  n_class]
        # dec_inputs: [batch_size, n_step,  n_class]
        B = len(enc_inputs)
        enc_inputs = self.embed(enc_inputs)
        dec_inputs = self.embed(dec_inputs)

        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs: [n_step, batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs: [n_step, batch_size, n_class]

        # enc_outputs : [n_step,      batch_size, n_hidden], matrix F
        # enc_hidden  : [num_layers , batch_size, n_hidden]
        enc_outputs, _ = self.enc_cell(enc_inputs, hidden)
        enc_hidden = enc_outputs[-1:]

        # trained_attn = []
        hidden       = enc_hidden
        n_step       = len(dec_inputs)
        model_hidden = torch.zeros([n_step, B, 2*self.n_hidden],device=self.device)
        trained_attn = torch.zeros([B,   n_step,  len(enc_inputs)],device=self.device)
        # model_output = torch.empty([n_step, 1, n_class])

        for i in range(n_step):
            # each time step
            # dec_output : [n_step(=1),      batch_size(=1),  n_hidden]
            # hidden     : [num_layers(=1) , batch_size(=1),  n_hidden]
            # attn_weights : [1, 1, n_step]

            #### Teacher forcing decoding
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            attn_weights       = self.get_att_weight(dec_output, enc_outputs)
            # bmm: batched matrix multiplication
            # [1,1,n_step] x [1,n_step,n_hidden] = [1,1,n_hidden]
            context            = attn_weights.bmm(enc_outputs.transpose(0, 1)).transpose(0,1)
            model_hidden[i]    = torch.cat((dec_output, context), 2)

            trained_attn[:,i]  = (attn_weights.squeeze().detach())

            # import pdb; pdb.set_trace()
        output_logit = self.out_layer(model_hidden)
        output_logit = output_logit.transpose(0, 1)

        # make model shape [n_step, n_class]
        return output_logit, trained_attn.transpose(2,1)

    def get_att_weight(self, dec_output, enc_outputs):  # get attention weight one 'dec_output' with 'enc_outputs'
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step,device=self.device)  # attn_scores : [n_step]

        enc_t = self.attn(enc_outputs)
        score = dec_output.transpose(1,0).bmm(enc_t.transpose(1,0).transpose(2,1))
        out1   = score.softmax(-1)
        return out1

class Seq2SeqWithNoAttention(Seq2SeqWithAttention):
    def _forward(self, enc_inputs, hidden, dec_inputs):
        # enc_inputs: [batch_size, n_step,  n_class]
        # dec_inputs: [batch_size, n_step,  n_class]
        B = len(enc_inputs)
        enc_inputs = self.embed(enc_inputs)
        dec_inputs = self.embed(dec_inputs)

        enc_inputs = enc_inputs.transpose(0, 1)  # enc_inputs: [n_step, batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)  # dec_inputs: [n_step, batch_size, n_class]

        # enc_outputs : [n_step,      batch_size, n_hidden], matrix F
        # enc_hidden  : [num_layers , batch_size, n_hidden]
        enc_outputs, _ = self.enc_cell(enc_inputs, hidden)
        enc_hidden = enc_outputs[-1:]

        # trained_attn = []
        hidden       = enc_hidden
        n_step       = len(dec_inputs)
        model_hidden = torch.zeros([n_step, B, 2*self.n_hidden],device=self.device)
        trained_attn = torch.zeros([B,   n_step,  len(enc_inputs)],device=self.device)
        # model_output = torch.empty([n_step, 1, n_class])

        for i in range(n_step):
            # each time step
            # dec_output : [n_step(=1),      batch_size(=1),  n_hidden]
            # hidden     : [num_layers(=1) , batch_size(=1),  n_hidden]
            # attn_weights : [1, 1, n_step]

            #### Teacher forcing decoding
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            # attn_weights       = self.get_att_weight(dec_output, enc_outputs)
            # bmm: batched matrix multiplication
            # [1,1,n_step] x [1,n_step,n_hidden] = [1,1,n_hidden]
            # context            = attn_weights.bmm(enc_outputs.transpose(0, 1)).transpose(0,1)
            model_hidden[i]    = torch.cat((dec_output, dec_output), 2)

            # trained_attn[:,i]  = (attn_weights.squeeze().detach())

            # import pdb; pdb.set_trace()
        output_logit = self.out_layer(model_hidden)
        output_logit = output_logit.transpose(0, 1)

        # make model shape [n_step, n_class]
        return output_logit, trained_attn



def main():
    from markov_lm.Model_NLP import NLPLayerConfig

    # n_step = 5 # number of cells(= number of Step)
    # n_hidden = 128 # number of hidden units in one cell
    CUDA = 1

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    # ax.matshow((trained_attn[0]).detach().cpu().numpy(), cmap='viridis')

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    # n_class = len(word_dict)  # vocab list


    # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
    device = torch.device('cuda:0' if CUDA else 'cpu')
    model = Seq2SeqWithAttention(config=NLPLayerConfig(embed_dim=128,model_name='Seq2Seq_Attention',graph_dim=len(word_dict)),device=device )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    self = model
    hidden = torch.zeros((1, 1, model.n_hidden)).to(self.device)

    def make_batch():
        n_class = model.n_class
        # input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
        # output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]

        # return (torch.tensor(input_batch,device=self.device).float(),
        # torch.tensor(output_batch,device=self.device).float(),
        # torch.tensor(target_batch,device=self.device).long())

        input_batch = [[word_dict[n] for n in sentences[0].split()]]
        output_batch = [[word_dict[n] for n in sentences[1].split()]]
        target_batch = [[word_dict[n] for n in sentences[2].split()]]

        return (torch.tensor(input_batch,device=self.device).long(),
        torch.tensor(output_batch,device=self.device).long(),
        torch.tensor(target_batch,device=self.device).long())

    input_batch, output_batch, target_batch = make_batch()
    # Train
    for epoch in range(2000):
        optimizer.zero_grad()

        # output, _ = model.forward(input_batch, hidden, output_batch)
        # loss = -torch.gather(output.log_softmax(-1),index=target_batch.unsqueeze(-1),dim=-1)
        # loss = loss.mean()

        loss = model.loss(dict(source=input_batch,target=output_batch))

        # output, _ = model._forward(input_batch, hidden, output_batch)
        # loss = criterion(output.squeeze(0), target_batch.squeeze(0))
        #
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    # test_batch = [np.eye(self.n_class)[[word_dict[n] for n in 'SPPPP']]]
    # test_batch = torch.FloatTensor(test_batch).to(self.device)

    test_batch = [[word_dict[n] for n in 'SPPPP']]
    test_batch = torch.FloatTensor(test_batch).to(self.device).long()

    # predict,trained_attn = model.forward(dict(source=input_batch,target=test_batch))

    predict, trained_attn = model._forward(input_batch, hidden, test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    '''
    Epoch: 0400 cost = 0.000321
    Epoch: 0800 cost = 0.000112
    Epoch: 1200 cost = 0.000057
    Epoch: 1600 cost = 0.000034
    Epoch: 2000 cost = 0.000023
    ich mochte ein bier P -> ['i', 'want', 'a', 'i', 'i']
    '''
    # Show Attention
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow((trained_attn[0]).detach().cpu().numpy(), cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    # plt.show()
    with open(__file__+'.html','w') as f:
        f.write(write_png_tag(plt.gcf()))

if __name__ == '__main__':
    main()
