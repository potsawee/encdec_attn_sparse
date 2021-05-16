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
    batch_size     = config['batch_size']
    lr0            = config['lr0']
    warmup         = config['warmup']
    temperature    = config['temperature']
    eps            = config['eps']
    gradient_accum = config['gradient_accum']
    valid_step     = config['valid_step']
    total_step     = config['total_step']
    # early_stop     = config['early_stop'] # currently we validation script to do this separately -- save time in training
    random_seed    = config['random_seed']
    task           = config['task'] # CNNDM | XSUM
    max_target_len = config['max_target_len']
    num_encoder_sent_nn_layers = config['num_encoder_sent_nn_layers']

    bart_tokenizer = BartTokenizer.from_pretrained(config['bart_tokenizer'])
    bart = BartEfficientDecoder.from_pretrained(config['bart_weights'])
    bart.sent_nn_init(num_encoder_sent_nn_layers)
    bart.swap_crossattn_to_hier(num_layers)

    if config['load_model_path'] != 'None':
        load_model_path = config['load_model_path']
        if torch_device == 'cuda': state = torch.load(load_model_path)
        else: state = torch.load(load_model_path, map_location=torch.device('cpu'))
        model_state_dict = state['model']
        bart.load_state_dict(model_state_dict)
        print('loaded model from:', load_model_path)
    else:
        print("integrated training starts from scracth!")

    if torch_device == 'cuda': bart.cuda()

    for p in bart.parameters(): p.requires_grad = False
    for p in bart.encoder_sent_nn.parameters(): p.requires_grad = True
    for p in bart.model.decoder.layers.parameters(): p.requires_grad = True
    # for key, p in bart.named_parameters():
        # if "sent_q" in key or "sent_k" in key:  p.requires_grad = True

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
        print("cnndm data loaded")
    elif task == 'XSUM':
        train_data = load_dataset('xsum', split='train')
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
    criterion    = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    criterion_kl = nn.KLDivLoss(reduction='mean')

    lambda1            = config['lambda1']
    num_sentences_kept = config['r_train']
    training_ref       = config['training_ref'] # exact, approx, mix
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
            # inspired by scheduled sampling
            use_exact_scores_prob = max(0.0, 1.0-(training_step/epoch_size))
            sampled_prob          = np.random.uniform(0,1)
            if sampled_prob < use_exact_scores_prob:
                boundary['use_exact_scores'] = True
            else:
                boundary['use_exact_scores'] = False
        else:
            raise Exception("training_ref not supported, please use exact|approx|mix")

        boundary['num_sentences_kept'] = num_sentences_kept
        bart_outputs = bart(input_ids=input_ids, decoder_input_ids=target_ids, boundary=boundary)
        # bart_outputs[0] --- predictive distribution (batch_size, target_len, vocab_size)
        # bart_outputs[1] --- decoder cross attention  [(batch_size, num_heads, target_len, num_sent)] x num_layers
        # bart_outputs[2] --- encoder outputs (batch_size, input_len, d)
        # bart_outputs[3] --- encoder self-attention  [(batch_size, num_heads, input_len, input_len)] x num_layers
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
            print("loss_kl is inf: id = {}".format(i))
            training_step += 1
            continue

        if torch.isnan(loss_kl):
            print("loss_kl is NaN: id = {}".format(i))
            training_step += 1
            continue

        # loss_kl = loss_kl / num_layers

        # Main Loss (cross entropy)
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        loss = loss.mean()

        total_loss = loss + lambda1*loss_kl
        total_loss.backward()

        if training_step % gradient_accum == 0:
            adjust_lr(optimizer, training_step, lr0, warmup)
            optimizer.step()
            optimizer.zero_grad()

        if training_step % 1 == 0:
            print("[{}] step {}/{}: loss = {:.6f} | loss_kl = {:.6f}".format(str(datetime.now()), training_step, total_step, loss, loss_kl))
            sys.stdout.flush()

        if training_step % valid_step == 0:
            state = {
                'training_step': training_step,
                'model': bart.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            savepath = "{}/{}-step{}.pt".format(config['save_dir'], config['model_name'], training_step)
            torch.save(state, savepath)
            del state
            torch.cuda.empty_cache()
            print("Saved at {}".format(savepath))

        training_step += 1

    print("finish the experiment!")

if __name__ == "__main__":
    if(len(sys.argv) == 2):
        run_training(sys.argv[1])
    else:
        print("Usage: python train_KL.py config_path")
