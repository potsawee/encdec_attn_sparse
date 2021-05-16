import os
import sys
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from utils import parse_config, get_boundary_matrix_beamsearch
from transformers import BartTokenizer
from models.efficient_lobart_expA_decode_ideal import BartEfficientLoBART
name_config = parse_config("config", "name_config.py")

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch_device:", torch_device)

num_sentences_kept = 60
MODEL = 'arxiv_lobart_4k_orc'
DATA  = 'arxiv_mcs_4k'
MODEL_NAME = name_config[MODEL]
DATA_PATH  = name_config[DATA]


TRAINED_MODEL_PATH = "../../arxiv_sum0/lib/trained_models/{}.pt".format(MODEL_NAME)
DECODE_DIR         = "system_output/{}_{}/noapprox_num_sent_{}".format(MODEL, DATA, num_sentences_kept)

print("MODEL_NAME:", MODEL_NAME)
print("DATA_PATH:", DATA_PATH)
print("DECODE_DIR:", DECODE_DIR)

def experiment(start_id, end_id):
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    num_heads  = 16
    num_layers = 12

    attention_window = [1024]*12
    # LoBART
    bart = BartEfficientLoBART.from_pretrained('facebook/bart-large-cnn')
    bart.swap_selfattn_longformerattn(attention_window=attention_window)
    bart.expand_learned_embed_positions(multiple=4, cut=8)
    if torch_device == 'cuda':
        state = torch.load(TRAINED_MODEL_PATH)
    else:
        state = torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))
    model_state_dict = state['model']
    bart.load_state_dict(model_state_dict)

    # EfficientLoBART
    bart.swap_crossattn_to_hier(num_layers)
    if torch_device == 'cuda': bart.cuda()

    # data
    print("DATA_PATH =", DATA_PATH)
    with open(DATA_PATH, 'rb') as f:
        test_data = load_arxiv(DATA_PATH)
    print("len(test_data) = {}".format(len(test_data)))

    batch_size = 1
    assert batch_size == 1, "batch_size > 1 not supported"

    ids = [_i for _i in range(start_id, end_id)]
    random.shuffle(ids)

    bart.eval()

    for id in ids:
        outpath = "{}/{}_decoded.txt".format(DECODE_DIR, id)
        exist = os.path.isfile(outpath)
        if exist:
            print("id {}: already exists".format(id))
            continue

        # local-attention requires sequence length to be a multiple of ...
        input_text = " ".join(test_data[id].article_text)

        input_ids = bart_tokenizer.batch_encode_plus([input_text],
            return_tensors='pt', max_length=bart.config.max_position_embeddings,
            pad_to_max_length=True)['input_ids'].to(torch_device)

        # find boundaries
        _input_ids = input_ids[0].cpu().numpy()
        EOS_ID     = 479 # '.' with space
        num_beams  = 4
        boundary   = get_boundary_matrix_beamsearch(_input_ids, num_heads*batch_size, num_beams, EOS_ID, torch_device)
        boundary['num_sentences_kept'] = num_sentences_kept
        summary_ids = bart.generate(input_ids, boundary=boundary, num_beams=num_beams,
                        length_penalty=2.0, max_length=400, min_length=56,
                        no_repeat_ngram_size=3, pad_token_id=bart.config.pad_token_id)

        text = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True).strip()
        with open(outpath, 'w') as f:
            f.write(text)
        print("wrote:", outpath)

    print("finish the experiment!")

def load_arxiv(path):
    with open(path, 'rb') as f:
        articles = pickle.load(f, encoding="bytes")
    print("loaded:", path)
    return articles

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 100
        if end_id > 2000: end_idx = 2000
        experiment(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        experiment(start_id, end_id)
    else:
        print("Usage: python experiment.py start_id end_id")
        raise Exception("argv error")
