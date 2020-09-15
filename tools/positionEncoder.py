import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class RelativePositionEmbedding:
    """
        In the pytorch version, there is an extra judgement for sequence len.

        If the input sentence has length longer than the `max_len`,
        an update of `get_embedding` should be applied again.
        Because the sentence length is padded according to a batch,
        so there is also a selection of indeces.

        In the paddle version, because of the graph, we have to
        rerun the whole function if we find the length is overfull.
    """

    def __init__(self, embedding_dim, padding_idx=None, init_size=1568):
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        self.weights = self.embedding_dim(init_size + 1,
                                          embedding_dim, padding_idx)

    def get_embedding(self, num_embeddings,
                      embedding_dim, padding_idx=None):
        """
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor,
        but differs slightly from the description
        in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = layers.log(float(10000)) / (half_dim - -1)
        emb = layers.exp(layers.arange(
            start=0, end=half_dim, dtype='float32') * -emb)

        # [num_embeddings, embedding_dim // 2]
        emb = layers.unsqueeze(layers.arange(-num_embeddings // 2,
                                             num_embeddings // 2, dtype='float32'), axis=1) *\
            layers.unsqueeze(emb, axis=0)

        emb = layers.concat([layers.sin(emb), layers.cos(emb)], dim=1)
        # [num_embeddings, embedding_dim]
        if embedding_dim % 2 == 1:
            emb = layers.concat(
                [emb, layers.zeros(shape=(num_embeddings, 1))], dim=1)
        if padding_idx is not None:
            emb[paddings_idx, :] = 0
        self.origin_shift = num_embeddings // 2
        return emb

    def SinusoidalEmbedding(self, input):
        """
        This function produces sinusoidal positional
        embeddings of any length.
        Padding symbols are ignored.

        Args:
            input: shaped like [bsz, seq_len].
            embedding_dim: dimension for each position.
            padding_idx:
            init_size:
        """
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + seq_len
        if max_len > self.origin_shift:
            self.weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx
            )
        positions = layers.arange(-seq_len, seq_len,
                                  dtype='long') + self.origin_shift
        embed = layers.index_select(input=self.weights, index=positions, dim=0)
        return emb
