import numpy as np
import torch
import configparser
import collections

def get_boundary_matrix(input_ids: np.ndarray,
                        num_heads: int = 16,
                        eos_id: int = 4,
                        torch_device: str = 'cuda') -> torch.Tensor:
    # EOS_ID      = 4 # '.'
    eos_id2     = 2  #'</s>'
    input_w_len = input_ids.shape[0]
    input_s_len = int(np.array((input_ids == eos_id), dtype=np.float32).sum(axis=-1))
    input_s_len += int(np.array((input_ids == eos_id2), dtype=np.float32).sum(axis=-1))
    boundary = np.zeros((input_w_len, input_s_len), dtype=np.float32)
    sent_counter = 0
    word_counter = 0
    sentid2wordid = []
    wordid = []
    for j in range(input_w_len):
        if input_ids[j] == eos_id or input_ids[j] == eos_id2:
            num_word = word_counter + 1
            boundary[j-word_counter:j+1, sent_counter] = 1

            wordid.append(j)
            sentid2wordid.append(wordid)
            wordid = []

            sent_counter += 1
            word_counter = 0
        else:
            wordid.append(j)
            word_counter += 1
    boundary = np.tile(boundary, (num_heads, 1, 1))
    boundary = torch.tensor(boundary, device=torch_device, dtype=torch.float32)
    return {'boundary': boundary, 'sentid2wordid': sentid2wordid}


def shifted_target_left(target_ids):
    # shifted LEFT
    shifted_target_ids = torch.zeros(target_ids.shape, dtype=target_ids.dtype, device=target_ids.device)
    shifted_target_ids[:,:-1] = target_ids.clone().detach()[:,1:]
    return shifted_target_ids

def adjust_lr(optimizer, step, lr0, warmup):
    """to adjust the learning rate"""
    step = step + 1 # plus 1 to avoid ZeroDivisionError
    lr = lr0 * min(step**(-0.5), step*(warmup**(-1.5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return


def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif is_int(value):
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config

def print_config(config):
    print("######## Config ########")
    for key, value in config.items():
        print("{}: {}".format(key, value))
    print("########################")
    return
