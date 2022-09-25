'''
Source: https://github.com/nadavbh12/Character-Level-Language-Modeling-with-Deeper-Self-Attention-pytorch
license: MIT

'''
from markov_lm.external.clm_annotated_attention import *


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout,
                 intermediate_layer_predictions=True, generator=None, max_sequence_len=512, force_prediction=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.add_positional_encoding = AddPositionalEncoding(size, max_sequence_len)
        self.norm = self.sublayer[0].norm

        self.size = size
        self.intermediate_layer_predictions = intermediate_layer_predictions
        self.force_prediction = force_prediction
        if intermediate_layer_predictions and self.training:
            self.classifier = copy.deepcopy(generator)

    def forward(self, x, mask):
        x = self.add_positional_encoding(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        # if self.force_prediction or (self.intermediate_layer_predictions and self.training):
        #     return x, self.classifier(self.norm(x))
        # else:
        return x, None


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n_layers, intermediate_layer_predictions=True):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        # enforce a prediction for the last layer
        self.layers[-1].force_prediction = True
        self.norm = LayerNorm(layer.size)
        self.intermediate_layer_predictions = intermediate_layer_predictions

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        intermediate_predictions = []
        for layer in self.layers:
            x, prediction = layer(x, mask)
            intermediate_predictions.append(prediction)
        return self.norm(x), intermediate_predictions


class MultiLayerCrossEntropy(nn.Module):
    def __init__(self, vocab_size, *args, **kwargs):
        super(MultiLayerCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, layer_outputs, target):
        total_loss = torch.zeros(1, dtype=layer_outputs[-1].dtype, device=layer_outputs[-1].device)
        n_layers_with_loss = 0
        for layer_output in layer_outputs:
            if layer_output is not None:
                # if True:
                if self.training:
                    loss = self.cross_entropy(layer_output.view(-1, self.vocab_size).contiguous(), target)
                else:
                    # in evaluation consider only the last prediction
                    loss = self.cross_entropy(layer_output[:, -1, :].contiguous(), target)
                total_loss += loss
                n_layers_with_loss += 1

        average_loss_of_all_layers = total_loss / n_layers_with_loss
        final_layer_loss = loss
        return average_loss_of_all_layers, final_layer_loss


class NextCharTransformer(nn.Module):
    """
    A standard next-character prediction model. Base for this and many
    other models.
    """
    def __init__(self, vocab_size, n_layers=64,
                 hidden_size=512, inner_linear=2048,
                 n_heads=8, dropout=0.55, tied=True, max_sequence_len=512,
                 intermediate_layer_predictions=True):
        super(NextCharTransformer, self).__init__()

        attn = MultiHeadedAttention(n_heads, hidden_size, dropout)
        ff = PositionwiseFeedForward(hidden_size, inner_linear, dropout)

        generator = Generator(hidden_size, vocab_size)
        self.encoder = Encoder(EncoderLayer(hidden_size, copy.deepcopy(attn), copy.deepcopy(ff),
                                            dropout, intermediate_layer_predictions, generator,
                                            max_sequence_len),
                               n_layers, intermediate_layer_predictions)
        self.embed = Embeddings(hidden_size, vocab_size)

        self.criterion = MultiLayerCrossEntropy(vocab_size)

        # use weight sharing
        if tied:
            self.generator.proj.weight = self.src_embed.lut.weight

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        self.vocab_size = vocab_size
        self.intermediate_layer_predictions = intermediate_layer_predictions
        self.n_layers = n_layers

    def forward(self, src, mask):
        """Take in and process masked src and target sequences."""
        # import pdb; pdb.set_trace()
        src_emb = self.embed(src)
        emb, intermediate_predictions = self.encoder(src_emb, mask)
        return intermediate_predictions

    def update(self, training_percent):
        """Stop using losses from intermediate layer as function of time in training.
           See section 2.1 - Intermediate Layer Losses
        """
        for i, layer in enumerate(self.encoder.layers[:-1]):
            if training_percent > (i // (2 * self.n_layers)):
                layer.intermediate_layer_predictions = False


def next_char_transformer(src_vocab, n_layers=64, hidden_size=512,
                          inner_linear=2048, n_heads=8, dropout=0.55,
                          tied=True, max_sequence_len=512, intermediate_losses=True):
    return NextCharTransformer(src_vocab,
                               n_layers, hidden_size,
                               inner_linear, n_heads,
                               dropout, tied, max_sequence_len, intermediate_losses)
