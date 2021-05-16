import os
import sys
import random
from datetime import datetime
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_boundary_matrix, shifted_target_left, adjust_lr
from utils import parse_config, print_config
from data.loader import BartBatcher, load_podcast_cheatv3_4110_filtered_data
from data.processor import PodcastEpisode
from transformers import BartTokenizer
from models.efficient_lobart_expC_integrated import BartEfficientLoBART

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
    gradient_accum = config['gradient_accum']
    valid_step     = config['valid_step']
    total_step     = config['total_step']
    early_stop     = config['early_stop']
    random_seed    = config['random_seed']
    max_target_len = config['max_target_len']
    eps         = config['eps']
    temperature = config['temperature']

    bart_tokenizer = BartTokenizer.from_pretrained(config['bart_tokenizer'])
    # LoBART
    bart = BartEfficientLoBART.from_pretrained('facebook/bart-large-cnn')
    attention_window = [1024]*12
    bart.swap_selfattn_longformerattn(attention_window=attention_window)
    bart.expand_learned_embed_positions(multiple=4, cut=8)
    # EfficientLoBART
    bart.swap_crossattn_to_hier(num_layers)

    lobart_base_path  = config['lobart_base_path']
    if torch_device == 'cuda':
        bart.cuda()
        state = torch.load(lobart_base_path)
        model_state_dict = state['model']
        for key, val in bart.encoder_sent_nn.named_parameters():
            model_state_dict['encoder_sent_nn.'+key] = val
        for key, val in bart.named_parameters():
            if "sent_q" in key or "sent_k" in key:
                model_state_dict[key] = model_state_dict[key.replace("sent_","")]
        bart.load_state_dict(model_state_dict)
    else:
        raise Exception("CUDA device not detected")
    print('loaded model from:', lobart_base_path)
    print(bart)

    for p in bart.parameters(): p.requires_grad = False
    for p in bart.encoder_sent_nn.parameters(): p.requires_grad = True
    for p in bart.model.decoder.layers.parameters(): p.requires_grad = True

    print("#parameters:", sum(p.numel() for p in bart.parameters() if p.requires_grad))
    bart_config = bart.config

    bart.config.output_attentions = True
    bart.model.encoder.output_attentions = False
    bart.model.decoder.output_attentions = True
    for i in range(num_layers):
        bart.model.encoder.layers[i].output_attentions = False
        bart.model.decoder.layers[i].output_attentions = True

    # Data
    podcasts = load_podcast_cheatv3_4110_filtered_data(sets=-1) # -1 means set0,..,set9 (excluding 10)
    batcher = BartBatcher(bart_tokenizer, bart.model.config, podcasts, torch_device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, bart.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

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
    batcher.shuffle_podcasts()

    # by default, it's not training!!!
    bart.train()

    while training_step < total_step:
        input_ids, attention_mask, target_ids, target_attention_mask = batcher.get_a_batch(batch_size=batch_size, pad_to_max_length=False)
        shifted_target_ids = shifted_target_left(target_ids)

        # find boundaries
        _input_ids = input_ids[0].cpu().numpy()
        # boundary['boundary'].shape => [num_head, num_words, num_sent]
        boundary   = get_boundary_matrix(_input_ids, num_heads*batch_size, eos_id, torch_device)

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

        bart_outputs = bart(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=target_ids, boundary=boundary)
        # bart_outputs[0] --- predictive distribution (batch_size, target_len, vocab_size)
        # bart_outputs[1] --- decoder cross attention  [(batch_size, num_heads, target_len, num_sent)] x num_layers
        # bart_outputs[2] --- encoder outputs (batch_size, input_len, d)
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

        if training_step % valid_step == 0 and training_step > 0:
            state = {
                'training_step': training_step,
                'model': bart.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            savepath = "{}/{}-step{}.pt".format(SAVE_DIR, MODEL_NAME, training_step)
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
        print("Usage: python train_sparse.py config_path")
