import config
import torch
import transformers
import torch.nn as nn


def loss_fn(output, target, mask, num_labels):
    
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

def intent_loss_fn(output, target, mask, num_labels):
    
    lfn = nn.CrossEntropyLoss()
    
    active_logits = output.view(-1, num_labels)
    
    loss = lfn(active_logits, target.view(-1))
    return loss








class BERTIntExt(nn.Module):
    def __init__(self, num_tag, num_intents):
        super(BERTIntExt,self).__init__()
        self.num_tag = num_tag
        
        self.num_intents = num_intents
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.dropt_out_tag = nn.Dropout(0.3)
        
        self.dropt_out_intent = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768,self.num_tag)
        
        self.out_intent = nn.Linear(768,self.num_intents)
    

    def forward(self,ids,mask,token_type_ids,  target_tag,target_intent):
        out1, out2 = self.bert(ids,attention_mask=mask,token_type_ids = token_type_ids)

        bo_tag = self.dropt_out_tag(out1)
        
        bo_intent = self.dropt_out_intent(out2)

        tag = self.out_tag(bo_tag)
        
        intent = self.out_intent(bo_intent)

        loss_tag = loss_fn(tag,target_tag, mask, self.num_tag)
        
        loss_intent = intent_loss_fn(intent, target_intent, mask, self.num_intents)

        loss = (loss_tag +  loss_intent) / 2

        return tag,  intent, loss














