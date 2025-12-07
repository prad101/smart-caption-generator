import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

# from models.PositionalEncoding import PositionalEncoding

pd.set_option('display.max_colwidth', None)

# Hyperparameters
max_seq_len = 33
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Positional Encoding class
class PositionalEncoding(nn.Module):
    ''' 
    Implements positional encoding as described in "Attention is All You Need".
    Adds sinusoidal positional information to word embeddings to capture word order.
    This helps the Transformer model understand the sequence of words.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        

    def forward(self, x):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1).to(device)
        self.pe = self.pe[:x.size(0), : , : ]
        
        x = x + self.pe
        return self.dropout(x)

# Image Captioning Model using Transformer Decoder    
class ImageCaptionModel(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size):
        '''
        Initializes the image captioning Transformer model.
        - Creates word embedding layer
        - Adds positional encoding to capture word order
        - Builds Transformer Decoder layers
        - Adds final linear layer to predict next word
        - Initializes model weights
        '''
        super(ImageCaptionModel, self).__init__()
        positional_encoding = PositionalEncoding(embedding_size, 0.1)
        self.pos_encoder = positional_encoding
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model =  embedding_size, nhead = n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer = self.TransformerDecoderLayer, num_layers = n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        '''
        Initializes the model weights:
        - Embeddings and linear layer weights are set to small random values
        - Output layer bias is set to zero
        Helps stabilize training and prevents poor initial gradients.
        '''
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        '''
        Generates the masks required for Transformer decoding:
        1. Causal mask (decoder_input_mask):
           - Prevents the decoder from looking at future words
        2. Padding mask (decoder_input_pad_mask_bool):
           - Ignores <pad> tokens during attention
        Ensures fair, autoregressive caption generation.
        '''
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        '''
        Forward pass of the captioning model:
        Steps:
        1. Reshape encoded image features for Transformer
        2. Embed input caption tokens
        3. Add positional encoding to maintain word order
        4. Generate causal & padding masks for decoding
        5. Decode caption using image features + transformer decoder
        6. Apply final linear layer to predict the next word at each position

        Returns:
        - final_output: vocabulary logits for each time step
        - decoder_input_pad_mask: used later for masking loss
        '''
        encoded_image = encoded_image.permute(1,0,2)
        

        decoder_inp_embed = self.embedding(decoder_inp)* math.sqrt(self.embedding_size)
        
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)
        
        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)

        decoder_output = self.TransformerDecoder(tgt = decoder_inp_embed, memory = encoded_image, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)
        
        final_output = self.last_linear_layer(decoder_output)

        return final_output,  decoder_input_pad_mask