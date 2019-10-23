#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        e_char = 50 #character embedding size
        max_word_length = 21
        dropout_rate = 0.3
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embed_layer = nn.Embedding(len(vocab.char2id), e_char, padding_idx = pad_token_idx)
        self.cnn_layer = CNN(max_word_length, e_char, self.embed_size)
        self.highway_layer = Highway(self.embed_size, dropout_rate)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x_emb = self.char_embed_layer(input)
        x_emb = x_emb.permute(0, 1, 3, 2)
        x_word_emb = torch.zeros(x_emb.size()[0], x_emb.size()[1], self.embed_size, device = input.device) #seems not able to access self.device, so have to use input.device at here, not sure why
        for i in range(x_emb.size()[0]):
            x_conv_out = self.cnn_layer(x_emb[i, ...])
            x_word_emb[i, ...] = self.highway_layer(x_conv_out)
        return x_word_emb
        ### END YOUR CODE

