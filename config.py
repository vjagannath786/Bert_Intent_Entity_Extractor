import transformers
import torch

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 5
VALID_BATCH_SIZE = 5
EPOCHS = 5
BERT_PATH='bert-base-uncased'
MODEL_PATH='model.bin'
TRAINING_FILE =r'D:\ML\BERT_Multilingual_Intent_Entity_Model\input\customer_transfer_money_entities.csv'
TOKENIZER =transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case= True)


