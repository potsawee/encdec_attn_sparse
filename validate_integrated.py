import os
import sys
import random
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
from models.efficient_decoder_expC_integrated import BartEfficientDecoder
from utils import get_boundary_matrix, shifted_target_left
from utils import parse_config

def validate(config_path, model_path, cache_dir, start_id, end_id):
    # Load Config
    config = parse_config("config", config_path)

    # uses GPU in training or not
    if torch.cuda.is_available() and config['use_gpu']: torch_device = 'cuda'
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
    training_ref       = config['training_ref'] # exact, approx, mix
    num_sentences_kept = config['r_train']
    lambda1            = config['lambda1']

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
    criterion    = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    criterion_kl = nn.KLDivLoss(reduction='mean')

    assert batch_size == 1, "batch_size > 1 not supported"

    ids = [_i for _i in range(start_id, end_id)]
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

        shifted_target_ids = shifted_target_left(target_ids)

        # find boundaries
        _input_ids = input_ids[0].cpu().numpy()
        # boundary['boundary'].shape => [num_head, num_words, num_sent]
        boundary   = get_boundary_matrix(_input_ids, num_heads*batch_size, eos_id)

        if training_ref == 'exact':
            boundary['use_exact_scores'] = True
        elif training_ref == 'approx':
            boundary['use_exact_scores'] = False
        elif training_ref == 'mix':
            raise Exception("Change training_ref to exact or approx for validation!!")

        boundary['num_sentences_kept'] = num_sentences_kept

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

        # Main Loss (cross entropy)
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        loss = loss.mean().item()

        total_loss = loss + lambda1*loss_kl

        with open(outpath, 'w') as f:
            f.write("{:.8f}\n{:.8f}\n{:.8f}".format(total_loss, loss, loss_kl))
        print("wrote:", outpath)

if __name__ == "__main__":
    if(len(sys.argv) == 6):
        config_path = sys.argv[1]
        model_path  = sys.argv[2]
        cache_dir   = sys.argv[3]
        start_id    = int(sys.argv[4])
        end_id      = int(sys.argv[5])
        validate(config_path, model_path, cache_dir, start_id, end_id)
    else:
        print("Usage: python validate_integrated.py config_path model_path cache_dir start_id end_id")
