#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """
    Highway network of the word embedding
    """
    def __init__(self, e_word, dropout_rate = 0.5):
        """
        required parameters: 
            -e_word: size of the word embedding
        """
        super(Highway, self).__init__()
        self.e_word = e_word
        self.proj_layer = nn.Linear(e_word, e_word)
        self.gate_layer = nn.Linear(e_word, e_word)
        self.dropout_layer = nn.Dropout(p = dropout_rate)

    def forward(self, x):
        """
        forward process of the network
        size of x should be batch_size * e_word
        """
        x_proj = F.relu(self.proj_layer(x))
        x_gate = torch.sigmoid(self.gate_layer(x))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_proj
        x_word_emb = self.dropout_layer(x_highway)
        return x_word_emb
### END YOUR CODE 

