import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel


# Bert-BiGRU-Classifier
class BERTGRU(nn.Module):
    def __init__(self, Bio_BERT_PATH, padding_idx, bid=True, input_size=300, num_layers=2, hidden_size=300, dropout=0.5,nclasses=2):
        super(BERTGRU, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = BertModel.from_pretrained(Bio_BERT_PATH)
        self.num_layers = num_layers
        self.bid = bid
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nclasses = nclasses
        self.num_directions = (2 if self.bid else 1)
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size // self.num_directions,
            dropout=self.dropout,
            num_layers=self.num_layers,
            bidirectional=self.bid,
            batch_first=True,
        )

        self.fc_1 = nn.Linear(self.hidden_size, self.nclasses)
        self.d1 = nn.Dropout(self.dropout)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, cpos, ignore_cpos=False):
        # BERT
        mask = x != self.padding_idx
        #print(x.size())
        # print(x)
        with torch.no_grad():
            x = self.embedding(x)[0]
        # embedded, _ = self.embedding(tokens, attention_mask=masks)
        # cls_vector = embedded[:, 0, :]
        # cls_vector = cls_vector.view(-1, 1, self.input_size)

        # GRU
        #print(x.size())
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1), batch_first=True, enforce_sorted=False)
        #print(x.data.size())
        x, hidden = self.gru(x)
        #print(x.data.size())
        # hidden = hidden[-1]
        # print(hidden.squeeze(0).size())
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        #print(x.data.size())
        # Fully-connected layer
        # Get what we need
        row_indices = torch.arange(0, x.size(0)).long()
        #print(row_indices.size())
        # If this is  True we will always take the last state and not CPOS
        if ignore_cpos:
            x = hidden[0]
            x = x.view(self.num_layers, self.num_directions, -1, self.hidden_size//self.num_directions)
            x = x[-1, :, :, :].permute(1, 2, 0).reshape(-1, self.hidden_size)
        else:
            #print(x.size())
            x = x[row_indices, cpos, :]
        #print(x.size())

        # outputs = self.d1(hidden.squeeze(0))
        outputs = self.d1(x)
        outputs = self.fc_1(outputs)
        #print(outputs.size())
        # outputs = self.sigmoid(outputs)

        return outputs



# class BERTGRUSentiment(nn.Module):
#     def __init__(self,
#                  bert,
#                  hidden_dim,
#                  output_dim,
#                  n_layers,
#                  bidirectional,
#                  dropout):
        
#         super().__init__()
        
#         self.bert = bert
        
#         embedding_dim = bert.config.to_dict()['hidden_size']
        
#         self.rnn = nn.GRU(embedding_dim,
#                           hidden_dim,
#                           num_layers = n_layers,
#                           bidirectional = bidirectional,
#                           batch_first = True,
#                           dropout = 0 if n_layers < 2 else dropout)
        
#         self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text):
        
#         #text = [batch size, sent len]
                
#         with torch.no_grad():
#             embedded = self.bert(text)[0]
                
#         #embedded = [batch size, sent len, emb dim]
        
#         _, hidden = self.rnn(embedded)
        
#         #hidden = [n layers * n directions, batch size, emb dim]
        
#         if self.rnn.bidirectional:
#             hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
#         else:
#             hidden = self.dropout(hidden[-1,:,:])
                
#         #hidden = [batch size, hid dim]
        
#         output = self.out(hidden)
        
#         #output = [batch size, out dim]
        
#         return output
if __name__ == "__main__":
    Bio_BERT_PATH='/home/wanchu/MedCAT/biobert_large'
    embedding = BertModel.from_pretrained('distilroberta-base')
    print('first')
    gru = nn.GRU(
            input_size=100,
            hidden_size=300,
            dropout=0.5,
            num_layers=3,
            bidirectional=2,
            batch_first=True
        )
    print('second')
    l = nn.Linear(300, 1)
    print('third')