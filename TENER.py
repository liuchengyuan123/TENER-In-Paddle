import paddle
import paddle.fluid as fluid
from tools.embeddingMododels import PreEmbedding
from tools.positionEncoder import RelativePositionEmbedding
from tools.transformer import encoder


def TENER(word_input, char_input, word_vocab_size, char_vocab_size, embed_dim, num_layers, d_model,
          n_head, feed_forward_dim, attn_dropout, fc_dropout=0.3):
    """
    Intergrated TENER model.
    1. Embedding part.
    2. Multi-head attention part, including:
        1. layer normalization
        2. multi-head attention
        3. dropout layer
        4. residual connection and layer normalization
        5. positional feed forward layer
        6. dropout and residual
    3. Stacking layers.
    4. CRF classification.
    Args:
        word_input: Data holder for input words, data type `int`,
            refering the indexes of the words in the dictionary.
            Shaped like [batch_size, len_sentence].
        char_input: Similar to `word_input`, but for character level,
            and shaped like [batch_size, len_sentence, len_word].
        embed_dim: It could be a tuple or a list to set the embedding dim for
            word and character, which means [word_embed_dim, char_embed_dim].
            If it is a integer, char dim will be set to embed_dim // 2.
        num_layers: The number of layers in the third part (encoder) above.
        d_model: Dimension of the encoder outputs.
        n_head: Number of Attention heads.
        feed_forward_dim: Dimension of ffn.
        attn_dropout: Dropout in attention.
        fc_dropout: Dropout in (pre/post) process.

    """
    if hasattr(embed_dim, '__iter__'):
        assert (len(embed_dim) == 2), "Embedding Shape not understood."
        word_embed_dim = embed_dim[0]
        char_embed_dim = embed_dim[1]
    else:
        assert (isinstance(embed_dim, int)),\
            "Embed_dim should be an iterable object or integer."
        char_embed_dim = embed_dim // 2
        word_embed_dim = embed_dim - char_embed_dim

    total_embed = PreEmbedding(char_input, word_input, char_vocab_size,
                               word_vocab_size, char_embed_dim, word_embed_dim,
                               cnn_dim)
