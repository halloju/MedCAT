from meta_cat import MetaCAT
from tokenizers import ByteLevelBPETokenizer
import numpy as np

if __name__ == "__main__":
    tokenizer = ByteLevelBPETokenizer(vocab_file="/home/wanchu/MedCAT/medmen-vocab.json", merges_file="/home/wanchu/MedCAT/medmen-merges.txt")
    embeddings = np.load(open("/home/wanchu/MedCAT/embeddings.npy", 'rb'))
    mc = MetaCAT(embeddings=embeddings, tokenizer=tokenizer)
    mc.train("/home/wanchu/MedCAT/MedCAT_Export.json", 'Status', nepochs=20, Bio_BERT_PATH='/home/wanchu/MedCAT/biobert_large')