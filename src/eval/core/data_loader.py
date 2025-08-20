import re
from torch.utils.data import Subset
import torch

from core.config import SAMPLE_PATH, VAL_IDX_PATH
from core.reranker_dataset import TripletDataset
from transformers import AutoTokenizer

# Tokenizer configs
BASE_MODEL = 'bert-base-uncased'
MAX_LEN = 512
BATCH_SIZE = 16
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
def load_val_samples():
    dataset = TripletDataset(SAMPLE_PATH, tokenizer_name=BASE_MODEL, max_length=MAX_LEN)
    val_indices = torch.load(VAL_IDX_PATH)
    val_dataset = Subset(dataset, val_indices)
    return val_dataset

def parse_eval_pairs(text):
    # Match "query : <query text> title : <title text>"
    pattern = re.compile(r"query\s*:\s*(.*?)\s*title\s*:\s*(.*?)\s*genres", re.IGNORECASE | re.DOTALL)
    pairs = []
    for match in pattern.finditer(text):
        query = " ".join(match.group(1).split()).strip(" -")
        title = match.group(2).strip()
        # Capitalize first letter of each word in title except for small words
        pairs.append({"query": query, "positive": title})
    return pairs

def load_held_out_sample(num_sample):
    held_out =[]  
    val_dataset = load_val_samples()

    for i in range (num_sample):
        held_out.extend(parse_eval_pairs(tokenizer.decode(val_dataset[i]['input_ids_pos'], skip_special_tokens=True)))
    return held_out