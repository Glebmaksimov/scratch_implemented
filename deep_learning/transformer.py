import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    """
    d_model: size of the positional encoding vector per each token
    vocab_size: number of tokens in vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # TODO

    def forward(self, x):  # TODO
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)  # TODO (making model less overfit)

        # matrix of shape (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        # Vettor of shape (seq_len, 1) that represents position of the word inside the sentance
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        denominator = torch.exp(  # vector of shape (d_model)
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # TODO (check the formulas)

        # Apply sin(position * (10000 ** (2i / d_model)) to even positions
        positional_encoding[:, 0::2] = torch.sin()

        # Apply cos(position * (10000 ** (2i / d_model)) to odd  positions
        positional_encoding[:, 1::2] = torch.cos(position * denominator)

        # Add a batch dimension to the positional encoding (1, seq_len, d_model) to have a possibility to process batch of prompts.
        positional_encoding = positional_encoding.unsqueeze(0)

        # Register the positional encoding as a buffer to save it while model state saving.
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        # Summorizing positional encoding each word in given sentance (batch, seq_len, d_model)
        x += (self.positional_encoding[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    """
    Calculetes mean and variance for each prompt in batch independantly
    and calculates the new values for each prompt using it's mean and variance using $\hat x_j = \frac{x_j - \mu_j}{\sqrt{\theta_j^2 + \epsilon}}$
    """

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))  # learnable parameter

    def forward(self, x):  # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting  (batch, seq_len, 1) | mean and std methods cancel the dimention it was applied.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (
            self.alpha * (x - mean) / math.sqrt((std + self.eps)) + self.bias
        )  # Foemula was simplified to make less computation | we need eps to prevent dividing by zero or when std is very small


class FeedForwardBlock(nn.Module):  # TODO

    """
    dff: inner-layer dimensionality;
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # We have an input sentance tensor (batch, seq_len, d_model) -linear_1-> (batch, seq_len, d_ff) -linear_2-> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.heads = num_heads  # Number of heads
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        self.d_k = (
            d_model // num_heads
        )  # Dimension of embedding vector seen by each head (dk in shema)
        self.w_q = nn.Linear(
            d_model, d_model, bias=False
        )  # W^q weight matrics for Q with shape(d_model,d_model)
        self.w_k = nn.Linear(
            d_model, d_model, bias=False
        )  # W^k weight matrics for K with shape(d_model,d_model)
        self.w_v = nn.Linear(
            d_model, d_model, bias=False
        )  # W^v weight matrics for V with shape(d_model,d_model)

        self.w_o = nn.Linear(
            d_model, d_model, bias=False
        )  # W^o weight matrics  with shape(num_heads * dv,d_model) | num_heads * dv == d_model
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        We need mask to disable upper diagonal interactions(vectors).It is triangular matrix of ones.
        """

        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(
                mask == 0, -1e9
            )  # write a very low value (indicating -inf) to the positions where mask == 0

        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (
            attention_scores @ value
        ), attention_scores  # attention_scores will be used used for visualization

    def forward(self, q, k, v, mask):
        query = self.w_q(
            q
        )  # multiplication of Q marics by coresponding weight matrix (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(
            k
        )  # multiplication of K marics by coresponding weight matrix (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(
            v
        )  # multiplication of V marics by coresponding weight matrix (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Deviding each Q,K,V matrices by 4 smaller ones (batch, seq_len, d_model) -view-> (batch, seq_len, h, d_k) -transposition-> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Combine all the heads together
        # (batch, h, seq_len, d_k) -transposition-> (batch, seq_len, h, d_k) -view-> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by W^o | (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # 1 Multi-Head attension block
        self.feed_forward_block = feed_forward_block  # 1 FeedForward block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )  # 2 Add&Norm blocks

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )  # x,x,x <==> Q,K,V
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    layers:  list of EncoderBlocks
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    """
    We have two masks src_mask, tgt_mask because we have translation task, and src_mask for English and it came form Encoder and
    tgt_mask for Italian that came from the Decoder.
    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = (
            self_attention_block  # 1 Multi-Head Self Attention Block
        )
        self.cross_attention_block = cross_attention_block  # 1 Cross-Attention Block
        self.feed_forward_block = feed_forward_block  # 1 FeedForward Block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )  # 3 Add&Norm Blocks

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # proj -> projection

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(
            self.proj(x), dim=-1
        )  # log_softmax for numerical stability


class Transformer(nn.Module):

    """
    src - components for English.
    tgt = components for Italian.

    Here we do not use forward method because during inference we can reuse encoder output.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):  # Encoder Block
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):  # Decoder Block
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):  # Projection Block
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the weights not rundomly for faster trinining.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
