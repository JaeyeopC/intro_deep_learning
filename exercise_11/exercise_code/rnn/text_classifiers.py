import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }

        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################


        self.hidden_size = hidden_size
        self.num_embeddings = num_embeddings

        
        self.embedding = nn.Embedding(self.hparams["num_embeddings"], self.hparams["embedding_dim"], 0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, 1)
        # self.linear = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size, 1)
        self.Sigmoid = nn.Sigmoid()
        


        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################
        output = self.embedding(sequence)
        if lengths is not None:
            output = pack_padded_sequence(output, lengths)

        _, (hn, cn) = self.lstm(output)
        output = self.linear(hn) 

        output = self.Sigmoid(output)
        output = output.squeeze().view(-1)
        
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output
