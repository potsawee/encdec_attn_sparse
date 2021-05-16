import os
import sys
import random
import argparse
import numpy as np
import torch

from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

def decode(args):
    start_id   = args['start_id']
    end_id     = args['end_id']
    decode_dir = args['decode_dir']
    task       = args['dataset']

    # uses GPU in training or not
    if torch.cuda.is_available() and args['use_gpu']: torch_device = 'cuda'
    else: torch_device = 'cpu'

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    if task == 'CNNDM':  bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    elif task == 'XSUM': bart = BartEfficientDecoder.from_pretrained('facebook/bart-large-xsum')

    trained_model_path = args['load']
    if torch_device == 'cuda':
        bart.cuda()
        state = torch.load(trained_model_path)
    else:
        state = torch.load(trained_model_path, map_location=torch.device('cpu'))
    model_state_dict = state['model']
    bart.load_state_dict(model_state_dict)
    print('model loaded:', trained_model_path)
    bart.eval()

    if task == 'CNNDM':
        test_data = load_dataset('cnn_dailymail', '3.0.0', split='test')
        print("cnndm data loaded")
    elif task == 'XSUM':
        test_data = load_dataset('xsum', split='test')
        print("xsum data loaded")
    else:
        raise ValueError("task not supported, only CNNDM | XSUM")

    ids = [x for x in range(start_id, end_id)]
    if args['random_order']: random.shuffle(ids)

    # decoding hyperparameters
    num_beams = args['num_beams']
    length_penalty = args['length_penalty']
    max_length = args['max_length']
    min_length = args['min_length']
    no_repeat_ngram_size = args['no_repeat_ngram_size']

    for id in ids:
        outpath = "{}/{}_decoded.txt".format(decode_dir, id)
        exist = os.path.isfile(outpath)
        if exist:
            print("id {}: already exists".format(id))
            continue

        if task == 'CNNDM':  document = test_data[id]['article']
        elif task == 'XSUM': document = test_data[id]['document']

        batch_encoded_inputs = bart_tokenizer.batch_encode_plus([document], return_tensors='pt',
                            add_special_tokens=True, max_length=bart.config.max_position_embeddings, pad_to_max_length=False)
        input_ids      = batch_encoded_inputs['input_ids'].to(torch_device)
        attention_mask = batch_encoded_inputs['attention_mask'].to(torch_device)

        summary_ids = bart.generate(input_ids,
                        num_beams=num_beams, length_penalty=length_penalty,
                        max_length=max_length, min_length=min_length,
                        no_repeat_ngram_size=no_repeat_ngram_size)

        text = bart_tokenizer.decode(summary_ids[0].cpu().numpy(), skip_special_tokens=True)

        with open(outpath, 'w') as f:
            f.write(text)
        print("wrote:", outpath)

def get_decode_arguments(parser):
    '''Arguments for decoding'''

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # file paths
    parser.add_argument('--load',       type=str, required=True)  # path to load model
    parser.add_argument('--decode_dir', type=str, required=True)
    parser.add_argument('--dataset',    type=str, required=True)
    parser.add_argument('--start_id',   type=int, required=True)
    parser.add_argument('--end_id',     type=int, required=True)
    parser.add_argument('--num_beams',  type=int, default=4)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=50)
    parser.add_argument('--no_repeat_ngram_size',   type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=2.0)
    parser.add_argument('--random_order', type="bool", nargs="?", const=True, default=False)
    parser.add_argument('--use_gpu',    type="bool", nargs="?", const=True, default=False)

    return parser

if __name__ == "__main__":
    # get configurations from the terminal
    parser = argparse.ArgumentParser()
    parser = get_decode_arguments(parser)
    args = vars(parser.parse_args())
    decode(args)
