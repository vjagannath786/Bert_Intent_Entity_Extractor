import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import BERTIntExt



def process_data(data_path):
    df = pd.read_csv(data_path, encoding="utf-8")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    #print(df.head())

    enc_tag = preprocessing.LabelEncoder()

    enc_intents = preprocessing.LabelEncoder()
    

    
    df.loc[:, "tag"] = enc_tag.fit_transform(df["tag"])

    sentences = df.groupby("Sentence #")["word"].apply(list).values

    print(len(sentences)/2)
    
    tag = df.groupby("Sentence #")["tag"].apply(list).values

    _l = ['transfer']
    _b = ['balance']

    intents_1 = _l * int(len(sentences) / 2)
    intents_2 = _b * int(len(sentences) / 2)
    
    intents = intents_1 + intents_2
    print(intents)



    _intents = enc_intents.fit_transform(intents)


    return sentences, tag, enc_tag, _intents, enc_intents




if __name__ == "__main__":
    sentences, tag,  enc_tag, intents, enc_intents = process_data(config.TRAINING_FILE)

    
    
    meta_data = {
        
        "enc_tag": enc_tag,
        "enc_intents": enc_intents
    }

    joblib.dump(meta_data, "meta.bin")

    
    num_tag = len(list(enc_tag.classes_))
    num_intents = len(list(enc_intents.classes_))

    

    (
        train_sentences,
        test_sentences,
        
        train_tag,
        test_tag,
        train_intents,
        test_intents
    ) = model_selection.train_test_split(sentences, tag,intents,random_state=42, test_size=0.1)

    train_dataset = dataset.BERTDataset(
        texts=train_sentences, tags=train_tag, intents=train_intents
    )

    print(train_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        texts=test_sentences,  tags=test_tag, intents=test_intents
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)

    print(device)
    model = BERTIntExt(num_tag=num_tag, num_intents=num_intents)
    # #new code
    # print(len(config.TOKENIZER))  # 28996
    # config.TOKENIZER.add_tokens(["varun"])
    # print(len(config.TOKENIZER))  # 28997

    # config.TOKENIZER.save_pretrained('temp')
     
    # model.bert.resize_token_embeddings(len(config.TOKENIZER)) 
    # # The new vector is added at the end of the embedding matrix

    # print(model.bert.embeddings.word_embeddings.weight[-1, :])
    # # # Randomly generated matrix

    # model.bert.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.bert.config.hidden_size])

    # print(model.bert.embeddings.word_embeddings.weight[-1, :])

    # print(model.bert.embeddings.word_embeddings.weight.is_leaf)




    model.to(device)

    param_optimizer = list(model.named_parameters())
    #print(param_optimizer)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss

