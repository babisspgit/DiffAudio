import torch
import torch.nn as nn
import torch.nn.functional as F

#from transformers import BertForSequenceClassification
import math
from einops import rearrange


## Define the model
#class FakeRealClassifier(torch.nn.Module):
#    """ Classifier based on a pre-trained BERT model. """
#
#    def __init__(self, pretrained_model_name='bert-base-cased', num_labels=2):
#        super(FakeRealClassifier, self).__init__()
#        self.bert = BertForSequenceClassification.from_pretrained(
#            pretrained_model_name, num_labels=num_labels
#        )
#
#    def forward(self, input_ids, attention_mask):
#        """ Forward pass for the model. """
#        output = self.bert(input_ids, attention_mask=attention_mask)
#        return output.logits
#    
#    
#class TextTransformer(nn.Module):
#    """ Transformer model for text processing. """
#
#    def __init__(self, num_heads, num_blocks, embed_dims, vocab_size, 
#                 max_seq_len, num_classes=2, dropout=0.0):
#        super().__init__()
#        self.embedding = nn.Embedding(embedding_dim=embed_dims, num_embeddings=vocab_size)
#        self.positional_encoding = PositionalEncoding(embed_dims, max_seq_len)
#
#        encoder_blocks = [EncoderBlock(embed_dim=embed_dims, num_heads=num_heads) 
#                          for _ in range(num_blocks)]
#        self.text_transformer_blocks = nn.Sequential(*encoder_blocks)
#        self.output_layer = nn.Linear(embed_dims, num_classes)  # Output layer
#        self.dropout = nn.Dropout(dropout)
#
#    def forward(self, x):
#        """ Forward pass through the transformer. """
#        tokens = self.embedding(x)
#        x = self.positional_encoding(tokens)
#        x = self.dropout(x)
#        x = self.text_transformer_blocks(x)
#        x = x.max(dim=1)[0]  # Max pooling over sequence dimension
#        return self.output_layer(x)
#                        
#class Attention(nn.Module):
#    """ Multi-head attention mechanism. """
#
#    def __init__(self, num_heads, embed_dim):
#        super(Attention, self).__init__()
#        assert embed_dim % num_heads == 0, (
#            f"Embedding dimension ({embed_dim}) should be divisible "
#            f"by number of heads ({num_heads})"
#        )
#
#        self.num_heads = num_heads
#        self.embed_dim = embed_dim
#        self.head_dim = embed_dim // num_heads
#        self.scale = self.head_dim ** -0.5
#
#        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
#        self.k_projection = nn.Linear(embed_dim, embed_dim, bias=False)
#        self.v_projection = nn.Linear(embed_dim, embed_dim, bias=False)
#        self.o_projection = nn.Linear(embed_dim, embed_dim, bias=False)
#
#    def forward(self, x):
#        """ Forward pass for the attention mechanism. """
#        keys = self.k_projection(x)
#        queries = self.q_projection(x)
#        values = self.v_projection(x)
#
#        keys = rearrange(keys, "b seq (h d) -> (b h) seq d", h=self.num_heads)
#        values = rearrange(values, "b seq (h d) -> (b h) seq d", h=self.num_heads)
#        queries = rearrange(queries, "b seq (h d) -> (b h) seq d", h=self.num_heads)
#
#        attention_logits = torch.matmul(queries, keys.transpose(-2, -1))
#        attention_logits *= self.scale
#        attention = torch.nn.functional.softmax(attention_logits, dim=-1)
#        out = torch.matmul(attention, values)
#        out = rearrange(out, "(b h) seq d -> b seq (h d)", h=self.num_heads)
#        return self.o_projection(out)
#
#
#class EncoderBlock(nn.Module):
#    """ Encoder block in the transformer model. """
#
#    def __init__(self, embed_dim, num_heads, fc_hidden_dims=None, dropout=0.0):
#        super(EncoderBlock, self).__init__()
#        self.attention = Attention(num_heads, embed_dim)
#        self.layer_norm1 = nn.LayerNorm(embed_dim)
#        self.layer_norm2 = nn.LayerNorm(embed_dim)
#        self.dropout = nn.Dropout(dropout)
#
#        self.fc_hidden_dims = 4 * embed_dim if fc_hidden_dims is None else fc_hidden_dims
#        self.fc = nn.Sequential(
#            nn.LayerNorm(embed_dim),
#            nn.Linear(embed_dim, self.fc_hidden_dims),
#            nn.GELU(),
#            nn.LayerNorm(self.fc_hidden_dims),
#            nn.Linear(self.fc_hidden_dims, embed_dim),
#        )
#
#    def forward(self, x):
#        """ Forward pass for the encoder block. """
#        attention_output = self.attention(x)
#        x = self.layer_norm1(x + attention_output)
#        x = self.dropout(x)
#        fc_out = self.fc(x)
#        x = self.layer_norm2(fc_out + x)
#        x = self.dropout(x)
#        return x
#    
#class PositionalEncoding(nn.Module):
#    """ Positional encoding for transformer models. """
#
#    def __init__(self, embed_dim, max_seq_len=512):
#        super(PositionalEncoding, self).__init__()
#        pe = torch.zeros(max_seq_len, embed_dim)
#        position = torch.arange(0.0, max_seq_len).unsqueeze(1)
#        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
#        pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
#        pe = pe.unsqueeze(0)
#        self.register_buffer("pe", pe)
#
#    def forward(self, x):
#        """ Forward pass for positional encoding. """
#        return x + self.pe[:, :x.size(1)]
#    
#    
######################################################################################################
#
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, img_size=16, c_in=3, c_out=3, time_dim=256, device="gpu", channels=32):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, channels)
        self.down1 = Down(channels, channels*2,  emb_dim=time_dim)
        self.sa1 = SelfAttention(channels*2, img_size//2)
        self.down2 = Down(channels*2, channels*4, emb_dim=time_dim)
        
        self.sa2 = SelfAttention(channels*4, img_size // 4)
        self.down3 = Down(channels*4, channels*4,  emb_dim=time_dim)
        self.sa3 = SelfAttention(channels*4, img_size // 8)

        self.bot1 = DoubleConv(channels*4, channels*8)
        self.bot2 = DoubleConv(channels*8, channels*8)
        self.bot3 = DoubleConv(channels*8, channels*4)

        self.up1 = Up(channels*8, channels*2,  emb_dim=time_dim)
        self.sa4 = SelfAttention(channels*2, img_size // 4)
        self.up2 = Up(channels*4, channels,  emb_dim=time_dim)
        self.sa5 = SelfAttention(channels, img_size // 2)
        self.up3 = Up(channels*2, channels,  emb_dim=time_dim)
        self.sa6 = SelfAttention(channels, img_size)
        self.outc = nn.Conv2d(channels, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output