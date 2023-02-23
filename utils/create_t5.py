import re
import torch

import webdataset as wds
from tqdm import tqdm
import numpy as np
from utils.sequence_embedding import extract_sequence

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

T5_FEATURE_LENGTH = 1024
model_name = 'Rostlab/prot_t5_xl_bfd' # "Rostlab/prot_t5_xl_uniref50"

if "t5" in model_name:
  from transformers import T5EncoderModel, T5Tokenizer
  tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
  model = T5EncoderModel.from_pretrained(model_name)
elif "albert" in model_name:
  from transformers import AlbertModel, AlbertTokenizer
  tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = AlbertModel.from_pretrained(model_name)
elif "bert" in model_name:
  from transformers import BertModel, BertTokenizer
  tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = BertModel.from_pretrained(model_name)
elif "xlnet" in model_name:
  from transformers import XLNetModel, XLNetTokenizer
  tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = XLNetModel.from_pretrained(model_name)
else:
  print("Unkown model name")


# Remove any special tokens after embedding
if "t5" in model_name:
  shift_left = 0
  shift_right = -1
elif "bert" in model_name:
  shift_left = 1
  shift_right = -1
elif "xlnet" in model_name:
  shift_left = 0
  shift_right = -2
elif "albert" in model_name:
  shift_left = 1
  shift_right = -1
else:
  print("Unkown model name")

model = model.to(device)
model = model.eval()
if torch.cuda.is_available():
  model = model.half()

sequences_Example = ["A E T C Z A O","S K T Z P"]

sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

print(ids.keys())
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

with torch.no_grad():
    embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    # For feature extraction we recommend to use the encoder embedding
    encoder_embedding = embedding[0].cpu().numpy()[shift_left:shift_right]

print(encoder_embedding.shape)

url = "/data/wei/hpa-webdataset-all-composite/webdataset_info.tar"
dataset = wds.WebDataset(url).decode().to_tuple("__key__", "info.json")
with open("error-log-bert.txt", "w") as log:
    with wds.TarWriter('/data/wei/hpa-webdataset-all-composite/webdataset_t5.tar') as sink:
        for idx, data in tqdm(enumerate(dataset)):
            info = data[1]
            if info["sequences"]:
                try:
                    seq = extract_sequence(info["sequences"][0])
                    seq = " ".join(seq)
                    ids = tokenizer.batch_encode_plus([re.sub(r"[UZOB]", "X", seq)], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
                    input_ids = ids['input_ids'].to(device)
                    # attention_mask = ids['attention_mask'].to(device)
                    with torch.no_grad():
                        embedding = model(input_ids=input_ids)
                        # compute per_protein embedding
                        embedding = embedding.last_hidden_state[0].mean(dim=0)
                        t5_embedding = embedding.cpu().numpy()
                        assert t5_embedding.shape == (T5_FEATURE_LENGTH, )
                except Exception as e:
                    log.write(f"Failed to run bert for {info}\n")
                    print(e, info)
            else:
                t5_embedding = np.zeros([T5_FEATURE_LENGTH], dtype='float32')
            sink.write({
                "__key__": data[0],
                "t5.pyd": t5_embedding
            })
