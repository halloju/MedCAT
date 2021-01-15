from meta_cat import MetaCAT
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel
from transformers import BertTokenizer

if __name__ == "__main__":
    tokenizer = ByteLevelBPETokenizer(vocab_file="/home/wanchu/MedCAT/medmen-vocab.json", merges_file="/home/wanchu/MedCAT/medmen-merges.txt")
    # print(tokenizer)
    

    # tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    embeddings = np.load(open("/home/wanchu/MedCAT/embeddings.npy", 'rb'))
    mc = MetaCAT(embeddings=embeddings, tokenizer=tokenizer,pad_id=len(embeddings)-1)
    mc.train("/home/wanchu/MedCAT/MedCAT_Export.json", 'Status', nepochs=20, model_name='bert_gru')
    # from utils.BERTGRU import BERTGRU
    # Bio_BERT_PATH='/home/wanchu/MedCAT/biobert_large'
    # m = BERTGRU(Bio_BERT_PATH, nclasses=1, bid=2, num_layers=3,
    #                          input_size=300, hidden_size=300, dropout=0.3)
    # mc.save(full_save=True)

    # Bio_BERT_PATH='emilyalsentzer/Bio_ClinicalBERT'
    # embedding = BertModel.from_pretrained(Bio_BERT_PATH)
    # print('first')
    # gru = nn.GRU(
    #         input_size=100,
    #         hidden_size=300,
    #         dropout=0.5,
    #         num_layers=3,
    #         bidirectional=2,
    #         batch_first=True
    #     )
    # print('second')
    # l = nn.Linear(300, 1)
    # print('third')

    print('done')
#%%

# %%
