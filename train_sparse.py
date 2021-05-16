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
from models.efficient_decoder_expA import BartEfficientDecoder
from utils import get_boundary_matrix, shifted_target_left, adjust_lr
from utils import parse_config, print_config

def run_training(config_path):
    # Load Config
    config = parse_config("config", config_path)
    print_config(config)

    # uses GPU in training or not
    if torch.cuda.is_available() and config['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'

    num_heads      = config['num_heads']
    num_layers     = config['num_layers']
    eos_id         = config['eos_id']
    gamma          = config['gamma']
    batch_size     = config['batch_size']
    lr0            = config['lr0']
    warmup         = config['warmup']
    gradient_accum = config['gradient_accum']
    valid_step     = config['valid_step']
    total_step     = config['total_step']
    early_stop     = config['early_stop']
    random_seed    = config['random_seed']
    task           = config['task'] # CNNDM | XSUM
    max_target_len = config['max_target_len']

    bart_tokenizer = BartTokenizer.from_pretrained(config['bart_tokenizer'])
    bart = BartEfficientDecoder.from_pretrained(config['bart_weights'])
    bart.swap_crossattn_to_hier(num_layers)

    if torch_device == "cuda": bart.cuda()
    print("#parameters:", sum(p.numel() for p in bart.parameters() if p.requires_grad))
    bart_config = bart.config

    bart.config.output_attentions = True
    bart.model.encoder.output_attentions = True
    bart.model.decoder.output_attentions = True
    for i in range(num_layers):
        bart.model.encoder.layers[i].output_attentions = True
        bart.model.decoder.layers[i].output_attentions = True

    if task == 'CNNDM':
        train_data = load_dataset('cnn_dailymail', '3.0.0', split='train')
        valid_data = load_dataset('cnn_dailymail', '3.0.0', split='validation')
        print("cnndm data loaded")
    elif task == 'XSUM':
        train_data = load_dataset('xsum', split='train')
        valid_data = load_dataset('xsum', split='validation')
        print("xsum data loaded")
    else:
        raise ValueError("task not supported, only CNNDM | XSUM")

    # Optimizer --- currently only support Adam
    if config['optimizer'] == 'adam':
        # lr here doesn't matter as it will be changed by .adjust_lr()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, bart.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
        optimizer.zero_grad()
    else:
        raise ValueError("Current version only supports Adam")

    # Criterion
    criterion = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    training_step  = 0
    best_val_loss  = 9e9
    stop_counter   = 0

    # Randomness
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    assert batch_size == 1, "batch_size > 1 not supported"

    epoch_size    = len(train_data)
    print("epoch_size:", epoch_size)
    instance_ids  = [i for i in range(epoch_size)]
    random.shuffle(instance_ids)

    # by default, it's not training!!!
    bart.train()

    while training_step < total_step:
        # batch_size = 1
        if len(instance_ids) > 0:
            i = instance_ids.pop()
        else:
            instance_ids  = [i for i in range(epoch_size)]
            random.shuffle(instance_ids)
            i = instance_ids.pop()

        if task == 'CNNDM':
            document = train_data[i]['article']
            summary  = train_data[i]['highlights']
        elif task == 'XSUM':
            document = train_data[i]['document']
            summary  = train_data[i]['summary']

        try:
            # some training data instances are corrupted
            batch_encoded_inputs = bart_tokenizer.batch_encode_plus([document], return_tensors='pt',
                                add_special_tokens=True, max_length=bart.config.max_position_embeddings, pad_to_max_length=False)
            input_ids      = batch_encoded_inputs['input_ids'].to(torch_device)
            batch_encoded_target = bart_tokenizer.batch_encode_plus([summary], return_tensors='pt',
                                add_special_tokens=True, max_length=max_target_len, pad_to_max_length=False)
            target_ids     = batch_encoded_target['input_ids'].to(torch_device)
            shifted_target_ids = shifted_target_left(target_ids)
        except IndexError:
            print("[IndexError]: id = {}".format(i))
            continue

        # find boundaries
        _input_ids = input_ids[0].cpu().numpy()
        # boundary['boundary'].shape => [num_head, num_words, num_sent]
        boundary   = get_boundary_matrix(_input_ids, num_heads*batch_size, eos_id)

        bart_outputs = bart(input_ids=input_ids, decoder_input_ids=target_ids, boundary=boundary)
        # bart_outputs[0] --- predictive distribution (batch_size, target_len, vocab_size)
        # bart_outputs[1] --- decoder cross attention  [(batch_size, num_heads, target_len, num_sent)] x num_layers
        # bart_outputs[2] --- encoder outputs (batch_size, input_len, d)
        # bart_outputs[3] --- encoder self-attention  [(batch_size, num_heads, input_len, input_len)] x num_layers

        lm_logits               = bart_outputs[0]
        exact_sent_attn_weights = bart_outputs[1]

        # L_{seq2seq} --- Main Loss
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        loss = (loss).mean()

        # L_{sparsity} --- Computing Entropy
        if gamma > 0.0:
            loss_sparsity = 0
            for l in range(num_layers):
                # Since batch_size is 1 already
                prob = exact_sent_attn_weights[l][0] + 1e-12 # avoid 0*log(0)
                plogp = prob * torch.log(prob)
                Hp = -1 * plogp.sum(dim=-1).mean() # mean across all heads, all decoding timesteps
                loss_sparsity += Hp
            loss_sparsity /= num_layers
        else:
            loss_sparsity = 0

        # Total loss
        total_loss = loss + gamma * loss_sparsity
        total_loss.backward()


        if training_step % gradient_accum == 0:
            adjust_lr(optimizer, training_step, lr0, warmup)
            optimizer.step()
            optimizer.zero_grad()

        if training_step % 10 == 0:
            print("[{}] step {}/{}: loss = {:.3f} | loss_sparse = {:.3f}".format(str(datetime.now()), training_step, total_step, loss, loss_sparsity))
            sys.stdout.flush()

        if training_step % valid_step == 0 and training_step > 0:
            bart.eval()
            with torch.no_grad():
                valid_loss = validation(bart, bart_config, valid_data, batch_size, bart_tokenizer, task, max_target_len, torch_device)
            print("Valid Loss = {:.5f}".format(valid_loss))
            bart.train()
            if valid_loss < best_val_loss:
                stop_counter = 0
                best_val_loss = valid_loss
                print("Model improved".format(stop_counter))
            else:
                stop_counter += 1
                print("Model not improved #{}".format(stop_counter))
                if stop_counter == early_stop:
                    print("Stop training!")
                    return
            state = {
                'training_step': training_step,
                'model': bart.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            savepath = "{}/{}-step{}.pt".format(config['save_dir'], config['model_name'], training_step)
            torch.save(state, savepath)
            del state, valid_loss
            torch.cuda.empty_cache()
            print("Saved at {}".format(savepath))

        training_step += 1

    print("finish the experiment!")

def validation(bart, bart_config, valid_data, batch_size, bart_tokenizer, task, max_target_len, torch_device):
    print("start validating")
    criterion = nn.CrossEntropyLoss(reduction='none')
    sum_loss = 0
    sum_token = 0
    ids = [i for i in range(len(valid_data))]
    while len(ids) > 0:
        i = ids.pop()

        if task == 'CNNDM':
            document = valid_data[i]['article']
            summary  = valid_data[i]['highlights']
        elif task == 'XSUM':
            document = valid_data[i]['document']
            summary  = valid_data[i]['summary']

        try:
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
        x = bart(
            input_ids=input_ids,
            decoder_input_ids=target_ids,
        )
        lm_logits = x[0]
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        sum_loss  += (loss).sum().item()
        sum_token += loss.view(-1).size(0)
        print("#", end="")
        sys.stdout.flush()
    print()
    print("finish validating")
    return sum_loss / sum_token

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        run_training(sys.argv[1])
    else:
        print("Usage: python train_sparse.py config_path")
