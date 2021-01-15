import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
# from transformers import BertModel

class LSTM(nn.Module):
    def __init__(self, embeddings, padding_idx, nclasses=2, bid=True, input_size=300,
                 num_layers=2, hidden_size=300, dropout=0.5):
        super(LSTM, self).__init__()
        self.padding_idx = padding_idx
        # Get the required sizes
        vocab_size = len(embeddings)
        embedding_size = len(embeddings[0])
        print(embedding_size)

        self.num_layers = num_layers
        self.bid = bid
        self.input_size = input_size
        self.nclasses = nclasses
        self.num_directions = (2 if self.bid else 1)
        self.dropout = dropout

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embeddings.load_state_dict({'weight': embeddings})
        # Disable training for the embeddings - IMPORTANT
        self.embeddings.weight.requires_grad = False

        self.hidden_size = hidden_size
        bid = True # Is the network bidirectional

        # Create the RNN cell - devide 
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size // self.num_directions,
                           num_layers=self.num_layers,
                           dropout=dropout,
                           bidirectional=self.bid)
        self.fc1 = nn.Linear(self.hidden_size, nclasses)

        self.d1 = nn.Dropout(dropout)


    def forward(self, x, cpos, ignore_cpos=False):
        # Get the mask from x
        mask = x != self.padding_idx
        #print(x.size())
        # Embed the input: from id -> vec
        x = self.embeddings(x) # x.shape = batch_size x sequence_length x emb_size
        #print(x.size())
        # Tell RNN to ignore padding and set the batch_first to True
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1), batch_first=True, enforce_sorted=False)
        #print(x.data.size())
        # Run 'x' through the RNN
        x, hidden = self.rnn(x)
        #print(x.data.size())
        # Add the padding again
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #print(x.data.size())
        # Get what we need
        row_indices = torch.arange(0, x.size(0)).long()

        # If this is  True we will always take the last state and not CPOS
        if ignore_cpos:
            x = hidden[0]
            x = x.view(self.num_layers, self.num_directions, -1, self.hidden_size//self.num_directions)
            x = x[-1, :, :, :].permute(1, 2, 0).reshape(-1, self.hidden_size)
        else:
            #print(x.size())
            x = x[row_indices, cpos, :]
        #print(x.size())
        # Push x through the fc network and add dropout
        x = self.d1(x)
        #x = self.fc1(x)
        x = self.fc1(x)
        #print(x.size())
        return x



# Bert-BiGRU-Classifier
# class BERTGRU(nn.Module):
#     def __init__(self, Bio_BERT_PATH, bid=True, input_size=768, num_layers=5, hidden_size=768, dropout=0.5,nclasses=1):
#         super(BERTGRU, self).__init__()
#         self.embedding = BertModel.from_pretrained(Bio_BERT_PATH,from_tf=True)
#         self.num_layers = num_layers
#         self.bid = bid
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.nclasses = nclasses
#         self.num_directions = (2 if self.bid else 1)
#         self.dropout = dropout
#         self.gru = nn.GRU(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             dropout=self.dropout,
#             num_layers=self.num_layers,
#             bidirectional=self.bid,
#             batch_first=True,
#         )

#         self.fc_1 = nn.Linear(self.hidden_size, self.nclasses)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, tokens, masks=None, cpos=None, ignore_cpos=True):
#         # BERT
#         embedded, _ = self.embedding(tokens, attention_mask=masks)
#         cls_vector = embedded[:, 0, :]
#         cls_vector = cls_vector.view(-1, 1, self.input_size)

#         # GRU
#         _, hidden = self.gru(cls_vector)
#         hidden = hidden[-1]

#         # Fully-connected layer
#         outputs = self.fc_1(hidden.squeeze(0))
#         outputs = self.sigmoid(outputs).view(-1)

#         return outputs