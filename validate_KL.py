import os
import sys
import random
import argparse
from datetime import datetime
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# HuggingFace
from transformers import BartTokenizer
from datasets import load_dataset

# This project
from models.efficient_decoder_expC import BartEfficientDecoder
from utils import get_boundary_matrix
from utils import parse_config

def validate(args):
    config_path = args['config_path']
    model_path  = args['load']
    cache_dir   = args['cache_dir']
    start_id    = args['start_id']
    end_id      = args['end_id']

    # Load Config
    config = parse_config("config", config_path)

    # uses GPU in training or not
    if torch.cuda.is_available() and args['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'

    num_heads      = config['num_heads']
    num_layers     = config['num_layers']
    eos_id         = config['eos_id']
    batch_size     = config['batch_size']
    temperature    = config['temperature']
    eps            = config['eps']
    task           = config['task'] # CNNDM | XSUM
    max_target_len = config['max_target_len']
    num_encoder_sent_nn_layers = config['num_encoder_sent_nn_layers']

    bart_tokenizer = BartTokenizer.from_pretrained(config['bart_tokenizer'])
    bart = BartEfficientDecoder.from_pretrained(config['bart_weights'])
    bart.sent_nn_init(num_encoder_sent_nn_layers)
    bart.swap_crossattn_to_hier(num_layers)

    if torch_device == 'cuda': state = torch.load(model_path)
    else: state = torch.load(model_path, map_location=torch.device('cpu'))
    model_state_dict = state['model']
    bart.load_state_dict(model_state_dict)
    print('loaded model from:', model_path)

    if torch_device == 'cuda': bart.cuda()

    del state, model_state_dict
    torch.cuda.empty_cache()

    bart_config = bart.config
    bart.config.output_attentions = True
    bart.model.encoder.output_attentions = True
    bart.model.decoder.output_attentions = True
    for i in range(num_layers):
        bart.model.encoder.layers[i].output_attentions = True
        bart.model.decoder.layers[i].output_attentions = True

    if task == 'CNNDM':
        valid_data = load_dataset('cnn_dailymail', '3.0.0', split='validation')
        print("cnndm data loaded")
    elif task == 'XSUM':
        valid_data = load_dataset('xsum', split='validation')
        print("xsum data loaded")
    else:
        raise ValueError("task not supported, only CNNDM | XSUM")

    # Criterion
    criterion_kl = nn.KLDivLoss(reduction='mean')

    assert batch_size == 1, "batch_size > 1 not supported"

    ids = [_i for _i in range(start_id, end_id)]
    if args['random_order']: random.shuffle(ids)
    bart.eval()

    for id in ids:
        outpath = "{}/{}_vloss.txt".format(cache_dir, id)
        exist = os.path.isfile(outpath)
        if exist:
            print("id {}: already exists".format(id))
            continue

        if task == 'CNNDM':
            document = valid_data[i]['article']
            summary  = valid_data[i]['highlights']
        elif task == 'XSUM':
            document = valid_data[i]['document']
            summary  = valid_data[i]['summary']

        try:
            # some training data instances are corrupted
            batch_encoded_inputs = bart_tokenizer.batch_encode_plus([document], return_tensors='pt',
                                add_special_tokens=True, max_length=bart.config.max_position_embeddings, pad_to_max_length=False)
            input_ids      = batch_encoded_inputs['input_ids'].to(torch_device)
            batch_encoded_target = bart_tokenizer.batch_encode_plus([summary], return_tensors='pt',
                                add_special_tokens=True, max_length=max_target_len, pad_to_max_length=False)
            target_ids     = batch_encoded_target['input_ids'].to(torch_device)
        except IndexError:
            print("[IndexError]: id = {}".format(i))
            continue

        # find boundaries
        _input_ids = input_ids[0].cpu().numpy()
        boundary   = get_boundary_matrix(_input_ids, num_heads*batch_size, eos_id, torch_device)

        bart_outputs = bart(input_ids=input_ids, decoder_input_ids=target_ids, boundary=boundary)
        lm_logits  = bart_outputs[0]
        sent_attns = bart_outputs[1]

        num_sentences = len(boundary['sentid2wordid'])
        loss_kl = 0
        for l in range(num_layers):
            exact_attn = sent_attns[l]['exact_sent_attn_weights'].squeeze(0) # # (num_heads, tgt_len, num_sentences)
            exact_attn_hot = torch.softmax(torch.log(exact_attn + eps) / temperature, dim=-1)
            rnn_attn   = sent_attns[l]['rnn_sent_attn_weights'].squeeze(0) # # (num_heads, tgt_len, num_sentences)
            loss_kl += criterion_kl(torch.log(rnn_attn + eps).view(-1, num_sentences), exact_attn_hot.view(-1, num_sentences))

        if torch.isinf(loss_kl): # this works for both +ve and -ve inf
            print("loss_kl is inf: id = {}".format(id))
            raise("INF ERROR:", id)

        if torch.isnan(loss_kl):
            print("loss_kl is NaN: id = {}".format(id))
            raise("NAN ERROR:", id)

        loss_kl = loss_kl.item()

        with open(outpath, 'w') as f:
            f.write("{:.8f}".format(loss_kl))
        print("wrote:", outpath)

def get_decode_arguments(parser):
    '''Arguments for decoding'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # file paths
    parser.add_argument('--load',       type=str, required=True)  # path to load model
    parser.add_argument('--config_path',type=str, required=True)  # path to load model
    parser.add_argument('--cache_dir',  type=str, required=True)
    parser.add_argument('--start_id',   type=int, required=True)
    parser.add_argument('--end_id',     type=int, required=True)
    parser.add_argument('--random_order', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--use_gpu',    type="bool", nargs="?", const=True, default=False)
    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_decode_arguments(parser)
    args = vars(parser.parse_args())
    validate(args)
