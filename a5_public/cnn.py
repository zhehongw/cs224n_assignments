#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """
    CNN class used for word embedding
    """
    def __init__(self, m_word, e_char, kernel_size, e_word):
        """
        m_word: input layer max word lenth 
        e_char: input layer character embedding length
        kernel_size: convolution kernel size
        e_word: output channel size, the same as word embedding kernel_size
        """
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(e_char, e_word, kernel_size)
        self.max_pool_layer = nn.MaxPool1d(m_word - kernel_size + 1)

    def forward(self, input):
        x_conv = F.relu(self.conv_layer(input))
        x_conv_out = self.max_pool_layer(x_conv)
        #squeeze is needed to remove the last dimension of size one
        return torch.squeeze(x_conv_out)

### END YOUR CODE

