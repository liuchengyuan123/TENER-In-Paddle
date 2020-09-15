"""
Encoder part is inherited from
 https://github.com/PaddlePaddle/models/tree/release/1.8/PaddleNLP/machine_translation/transformer

Attention for the chage in `Scaled_dot_product`
"""

from functools import partial
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

dropout_seed = None


def wrap_layer_with_block(layer, block_idx):
    """
    Make layer define support indicating block, by which we can add layers
    to other blocks within current block. This will make it easy to define
    cache among while loop.
    """

    class BlockGuard(object):
        """
        BlockGuard class.

        BlockGuard class is used to switch to the given block in a program by
        using the Python `with` keyword.
        """

        def __init__(self, block_idx=None, main_program=None):
            self.main_program = fluid.default_main_program(
            ) if main_program is None else main_program
            self.old_block_idx = self.main_program.current_block().idx
            self.new_block_idx = block_idx

        def __enter__(self):
            self.main_program.current_block_idx = self.new_block_idx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.main_program.current_block_idx = self.old_block_idx
            if exc_type is not None:
                return False  # re-raise exception
            return True

    def layer_wrapper(*args, **kwargs):
        with BlockGuard(block_idx):
            return layer(*args, **kwargs)

    return layer_wrapper


def multi_head_attention(queries, keys, values, attn_bias, d_key, d_value, d_model, pos_enc,
                         n_head=1, dropout_rate=0., cache=None, static_kv=False):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activation to mask certain selected positions so that
    they will not considered in attention weights.

    Args:
        queries: input_sentence, shaped like [bsz, len_sentence, embedding_dim].
        keys: Most of the time, you just need queries, so set this value as None.
        values: Most of the time, you just need queries, so set this value as None.
        attn_bias: Bias added to the attention output before softmax,
            in case you want to mask some positions. Just set values as `inf`
            on these positions.
        d_key: The dimension wanted for keys and queries.
        d_value: The dimension wanted for values.
        d_model: output dimension of fully connected layer.
        pos_enc: Relative Positional encoder, whose shape is [2 X len_sentence, d_key].
        n_head: Number of attention heads.
        dropout_rate: probability on dropout layer.
    Return:
        The result of this multi-head attention layer.
        shape: [batch size, sentence len, d_model].
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors."
        )

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries, size=d_key * n_head,
                      bias_attr=False, num_flatten_dims=2)
        fc_layer = wrap_layer_with_block(
            layers.fc, fluid.default_main_program().current_block().parent_idx
        ) if cache is not None and static_kv else layers.fc
        k = fc_layer(input=keys, size=d_key * n_head,
                     bias_attr=False, num_flatten_dims=2)
        v = fc_layer(input=values, size=d_value * n_head,
                     bias_attr=False, num_flatten_dims=2)
        return q, k, v

    def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value)
    """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose. Specifically, transform the input tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
    # The value 0 in shape attr means copying the corresponding dimension
    # size of the input as the output dimension size.
    reshaped_q = layers.reshape(
        x=queries, shape=[0, 0, n_head, d_key], inplace=True)
    # permute the dimensions into:
    # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
    q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
    # For encoder-decoder attention in inference, insert the ops and vars
    # into global block to use as cache among beam search.
    reshape_layer = wrap_layer_with_block(
        layers.reshape,
        fluid.default_main_program().current_block(
        ).parent_idx) if cache is not None and static_kv else layers.reshape
    transpose_layer = wrap_layer_with_block(
        layers.transpose,
        fluid.default_main_program().current_block().
        parent_idx) if cache is not None and static_kv else layers.transpose
    reshaped_k = reshape_layer(
        x=keys, shape=[0, 0, n_head, d_key], inplace=True)
    k = transpose_layer(x=reshaped_k, perm=[0, 2, 1, 3])
    reshaped_v = reshape_layer(
        x=values, shape=[0, 0, n_head, d_value], inplace=True)
    v = transpose_layer(x=reshaped_v, perm=[0, 2, 1, 3])

    if cache is not None:  # only for faster inference
        cache_, i = cache
           if static_kv:  # For encoder-decoder attention in inference
                cache_k, cache_v = cache_["static_k"], cache_["static_v"]
                # To init the static_k and static_v in global block.
                static_cache_init = wrap_layer_with_block(
                    layers.assign,
                    fluid.default_main_program().current_block().parent_idx)
                static_cache_init(
                    k,
                    fluid.default_main_program().global_block().var(
                        "static_k_%d" % i))
                static_cache_init(
                    v,
                    fluid.default_main_program().global_block().var(
                        "static_v_%d" % i))
                k, v = cache_k, cache_v
            else:  # For decoder self-attention in inference
                # use cache and concat time steps.
                cache_k, cache_v = cache_["k"], cache_["v"]
                k = layers.concat([cache_k, k], axis=2)
                v = layers.concat([cache_v, v], axis=2)
                cache_["k"], cache_["v"] = (k, v)
        return q, k, v

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)
    
    def _shift(BD):
        """
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        to
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = layers.zeros(shape=(bsz, n_head, max_len, 1))
        BD = layers.reshape(x=layers.concat([BD, zero_pad], axis=-1),
            shape=(bsz, n_head, -1, max_len))
        BD = layers.reshape(x=BD[:, :, :-1], shape=(bsz, n_head, max_len, -1))
        BD = BD[:, :, :, max_len:]
        return BD

    def _transpose_shift(E):
        """
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

          to
          0  -10   -200
          1   00   -100
          2   10    000


        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = layers.zeros(shape=(bsz, n_head, max_len, 1))
        E = layers.reshape(x=layers.concat([E, zero_pad], axis=-1),
            shape=(bsz, n_head, -1, max_len))
        indice = layers.arange(start=0, end=max_len, dtype=int)
        E = layers.index_select(input=E, index=indice, dim=-2)
        E = layers.transpose(E, perm=[0, 1, 3, 2])
        return E

    def scaled_dot_product_attention(q, k, v, pos_enc, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention

        Change:
            - Different from the original one.
            We will remove the scale factor math: \sqrt{d_k} according to the paper.
            - Bias for attention and position encoding are added.
         
        """
        # product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_key**-0.5)

        # now q, k should be shaped like
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        # pos_enc should be shaped like [2 X l, head_dim], and head_dim = d_key
        max_sequence_len = q.shape[2]
        
        r_r_bias = layers.create_parameter(shape=(n_head, d_key))   # [n_head, head_dim]
        r_w_bias = layers.create_parameter(shape=(n_head, d_key))   # [n_head, head_dim]
        rw_head_q = q + r_r_bias[:, None]   # [batch, n_head, max_sequence_len, head_dim]
        AC = layers.matmul(x=rw_head_q, y=k, transpose_y=True)  # [batch, n_head, max_sequence_len, max_seqence_len]
        
        # position bias for each head, shaped like [n_head, 2 X max_sequence_len].
        # Then add two dimensions at `batch` and `maxlen`.
        D_ = layers.matmul(x=r_w_bias, y=pos_enc, transpose_y=True)[None, :, None]
        # position bias for each query, shaped like [batch, n_head, max_len, 2 X max_len]
        B_ = layers.matmul(x=q, y=pos_enc, transpose_y=True)
        # bias for each key, shaped like [batch, n_head, max_len, 2 X max_len]
        E_ = layers.matmul(x=k, y=pos_enc, transpose_y=True)
        
        # shaped like [batch, n_head, max_len, 2 X max_len]
        # change it to [batch, n_head, max_len, max_len]
        BD = B_ + D_
        BDE = _shift(BD) + _transpose_shift(E_)
        product = AC + BDE

        # product = layers.matmul(x=q, y=k, transposed_y=True, alpha=1.0) + \
        #     layers.matmul(x=q, y=pos_enc, transposed_y=True) +\
        #     layers.transpose(x=last_two, perm=[0, 1, 3, 2])
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                seed=dropout_seed,
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, pos_enc, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
    if dropout_rate:
        hidden = layers.dropout(
            hidden, dropout_prob=dropout_rate, seed=dropout_seed, is_test=False)
    out = layers.fc(input=hidden, size=d_hid, num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=dropout_seed,
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(enc_input, attn_bias, n_head, d_key,
                  d_value, d_model, d_inner_hid, pos_enc, prepostprocess_dropout,
                  attention_dropout, relu_dropout, preprocess_cmd='n',
                  postprocess_cmd='da'):
    """
    The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and dropout.

    Args:
        enc_input: Embedded input for the sentences. 
            (batch_size, len_sentence, embedding_dim)
        attn_bias: Bias added to the attention output before softmax,
            in case you want to mask some positions. Just set values as `inf`
            on these positions.
        n_head: Number of headers.
        d_key: Dimension of keys and queries.
        d_value: Dimension of values.
        d_model: Dimension of the encoder layer outputs.
        d_inner_hid: Dimension of the feed forward layer inside.
        pos_enc: Relative position encoder. (2 X max__len, d_key).
        prepostprocess_dropout: The dropout probability of the process layer
            before or after.
        attention_dropout: Dropout probability in the attention layer.
        relu_dropout: The activation in the feed forward layer is `relu`.
            Set the probability here.
        post/preprocess_cmd: The layers should be stacked. Use its default values
            unless the model needs to be changed.
    Return:
        An encoder layer output, (bsz, max_len, d_model).
    """
    attn_output = multi_head_attention(
        pre_process_layer(enc_input, preprocess_cmd, prepostprocess_dropout),
        None, None, attn_bias, d_key, d_value, d_model, pos_enc,
        n_head, attention_dropout
    )
    attn_output = post_process_layer(enc_input, attn_output,
                                     postprocess_cmd, prepostprocess_dropout)
    ffd_output = positionwise_feed_forward(
        pre_process_layer(attn_output, preprocess_cmd, prepostprocess_dropout),
        d_inner_hid, d_model, relu_dropout
    )
    return post_process_layer(attn_output, ffd_output,
        postprocess_cmd, prepostprocess_dropout)

def encoder(enc_input, attn_bias, n_layer, n_head,
    d_key, d_value, d_model, d_inner_hid, pos_enc,
    preporstprocess_dropout, attention_dropout,
    relu_dropout, preprocess_cmd='n',
    postprocess_cmd='da'):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.

    Args:
        enc_input: Embedded input for the sentences. 
            (batch_size, len_sentence, embedding_dim)
        attn_bias: Bias added to the attention output before softmax,
            in case you want to mask some positions. Just set values as `inf`
            on these positions.
        n_layers: Number of layers stacked together.
        n_head: Number of attention heads.
        d_key: Dimension of keys and queries.
        d_value: Dimension of values.
        d_model: Dimension of the encoder layer outputs.
        d_inner_hid: Dimension of the feed forward layer inside.
        pos_enc: Relative position encoder. (2 X max__len, d_key).
        prepostprocess_dropout: The dropout probability of the process layer
            before or after.
        attention_dropout: Dropout probability in the attention layer.
        relu_dropout: The activation in the feed forward layer is `relu`.
            Set the probability here.
        post/preprocess_cmd: The layers should be stacked. Use its default values
            unless the model needs to be changed.
    Return:
        Encoder output of the sentence input.
        (batch size, sentence len, d_model)
    """
    for i in range(n_layer):
        enc_output = encoder_layer(enc_input, attn_bias, n_head,
            d_key, d_value, d_model,d_inner_hid, pos_enc,
            prepostprocess_dropout, attention_dropout,relu_dropout,
            preprocess_cmd, postprocess_cmd
        )
        enc_input = enc_output
    enc_output = pre_process_layer(enc_output,
        preprocess_cmd, preporstprocess_dropout)
    return enc_output
