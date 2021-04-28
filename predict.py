import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import BERTIntExt



if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    
    enc_tag = meta_data["enc_tag"]
    enc_intents = meta_data["enc_intents"]

    
    num_tag = len(list(enc_tag.classes_))
    num_intents = len(list(enc_intents.classes_))

    sentence = """
    i want to transfer ₹100 to my daughter
    """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)
    tokenized_texts = [config.TOKENIZER.tokenize(sent) for sent in sentence]
    print(tokenized_texts)

    test_dataset = dataset.BERTDataset(
        texts=[sentence], 
        tags=[[0] * len(sentence)],
        intents = [[0]]
    )

    device = torch.device("cuda")
    model = BERTIntExt(num_tag=num_tag, num_intents=num_intents)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, intent, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )

        print(
            enc_intents.inverse_transform(
                intent.argmax(1).cpu().numpy().reshape(-1)
            )[0]
        )
        
