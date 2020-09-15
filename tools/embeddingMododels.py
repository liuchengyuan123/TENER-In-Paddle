import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def CNNCharEmbedding(input, vocab_size, cnn_dim, n_kernals,
                     hidden_dim, dropout_rate, output_dropout,
                     embed_dim):
    """
    CNN generates character embedding.
    Structed as:
    - embed(x)      len_word X hidden_dim
    - Dropout(x)    
    - CNN(x)        len_word X hidden_dim
    - activation(x)
    - pool          hidden_dim
    - fc            embed_dim
    - Dropout.
    Return:
     embedded Tensor shaped like [bsz, len_seq, embed_dim]
    """
    # input.size [batch_size, len_sentence, len_word]
    bsz, len_seq, len_word = input.shape
    emb = fluid.embedding(input, size=[vocab_size, hidden_dim])
    # emb.size [batch_size, len_sentence, len_word, hidden_dim]
    emb = layers.dropout(x=emb, dropout_prob=dropout_rate)
    emb = layers.reshape(x=emb, shape=(bsz * len_seq, 1, len_word, hidden_dim))
    # emb.size [batch_size X len_sentence, 1, len_word, hidden_dim]
    emb = layers.conv2d(input=emb, num_filters=n_kernals, filter_size=(
        cnn_dim, hidden_dim), padding=(cnn_dim-1, 0), act='relu')
    # emb.size [bsz X len_seq, n_kernals, len_word, 1]
    emb = layers.transpose(x=emb, perm=[0, 3, 2, 1])
    # emb.size [bsz X len_seq, 1, len_word, n_kernals]
    emb = layers.pool2d(input=emb, pool_size=[len_word, 1], pool_type='max')
    # emb.size [bsz X len_seq, 1, 1, n_kernals]
    emb = layers.fc(input=emb, size=embed_dim, num_flatten_dims=-1, act='tanh')
    # emb.size [bsz X len_seq, 1, 1, embed_dim]
    emb = layers.reshape(x=emb, shape=(bsz, len_seq, embed_dim))
    emb = layers.dropout(x=emb, dropout_prob=output_dropout)
    return emb


def WordEmbedding(input, vocab_size, embed_dim):
    """
    Word embedding, no pre-trained model used.
    Args:
     input: [bsz, len_seq].
     vocab_size: size of the vocabulary.
     embed_dim: word embedding_dim.
    Return:
     embeded tensor shaped like [bsz, len_seq, embed_dim].
    """
    emb = layers.embedding(input=input, size=[vocab_size, embed_dim])
    return emb


def PreEmbedding(char_input, word_input, char_vocab_size,
                 word_vocab_size, char_embed_dim, word_embed_dim, cnn_dim,
                 n_kernals, hidden_dim, output_dropout=0.1, dropout_rate=0):
    """
    An embedding function including both character level and word level.

    Args:
     char_input: place holder for character level input,
        shaped like [bsz, len_sentence, len_word].
     word_input: place holder for word level input, shaped like
        [bsz, len_sentence].
     char_vocab_size: int, size of character vocabulary.
     word_vocab_size: int, size of word vocabulary.
     char_embed_dim: character embedding dim.
     word_embed_dim: word embedding dim.
     cnn_dim: convolutional layer filter size in height.
     n_kernals: convolution kernal number.
     hidden_dim: first character embedding layer dim.
     output_dropout: prob of the last dropout layer before output.
     dropout_rate: word dropout probability.
    Return:
     input embedding, shaped like
         [bsz, len_sentence, char_embed_dim + word_embed_dim]
    """
    char_embed = CNNCharEmbedding(input=char_input, vocab_size=char_vocab_size,
                                  cnn_dim=cnn_dim, n_kernals=n_kernals,
                                  hidden_dim=hidden_dim, dropout_rate=dropout_rate,
                                  output_dropout=output_dropout, embed_dim=char_embed_dim)
    word_embed = WordEmbedding(input=word_input, vocab_size=word_vocab_size,
                               embed_dim=word_embed_dim)
    embeded = layers.concat(input=[char_embed, word_embed], axis=-1)
    return embeded
