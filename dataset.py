import config
import torch




class BERTDataset:
    def __init__(self, texts, tags, intents):
        self.texts = texts
        self.tags = tags
        
        self.intents = intents
    

    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, item):
        text = self.texts[item]
        
        tags = self.tags[item]
        intent = self.intents[item]

        ids =[]
        
        target_tag =[]

        for i, s in enumerate(text):

            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            
            target_tag.extend([tags[i]] * input_len)
        

        ids = ids[:config.MAX_LEN - 2]
        
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        
        target_tag = target_tag + ([0] * padding_len)

        target_intent = intent


        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "target_intent": torch.tensor(target_intent, dtype=torch.long),
        }
        



        
    


