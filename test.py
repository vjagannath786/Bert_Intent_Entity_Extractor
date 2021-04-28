# import pandas as pd
# import config
# from sklearn import preprocessing


# def process_data(data_path):
#     df = pd.read_csv(data_path, encoding="utf-8")
#     df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

#     #print(df.head())

#     enc_tag = preprocessing.LabelEncoder()

#     enc_intents = preprocessing.LabelEncoder()
    

    
#     df.loc[:, "tag"] = enc_tag.fit_transform(df["tag"])

#     sentences = df.groupby("Sentence #")["word"].apply(list).values
    
#     tag = df.groupby("Sentence #")["tag"].apply(list).values

#     _l = ['transfer']
#     _b = ['balance']

#     intents_1 = _l * (len(sentences) -10)
#     intents_2 = _b * (len(sentences) -10)
    
#     intents = intents_1 + intents_2
#     print(intents)

#     _intents = enc_intents.fit_transform(intents)


#     return sentences, tag, enc_tag, _intents, enc_intents


    
    




# sentences, tag, enc_tag, intents, enc_intents = process_data('D:\ML\BERT_Multilingual_Intent_Entity_Model\input\customer_transfer_money_entities.csv')


# print(intents)




import torch
from transformers import BertTokenizer, BertModel
#import config

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

print(len(tokenizer))  # 28996
tokenizer.add_tokens(["varun","₹"])
print(len(tokenizer))  # 28997

tokenizer.save_pretrained('temp')

model.resize_token_embeddings(len(tokenizer)) 
# The new vector is added at the end of the embedding matrix

print(model.embeddings.word_embeddings.weight[-1, :])
# Randomly generated matrix

#model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size], requires_grad=True, device='cuda')

#print(model.embeddings.word_embeddings.weight[-1, :])
# outputs a vector of zeros of shape [768]

#print(model.embeddings.word_embeddings.weight.is_leaf)