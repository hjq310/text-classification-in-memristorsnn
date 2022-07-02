import torch, torchdata, torchtext
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset(split_ratio, min_freq, BATCH_SIZE):
    # download the dataset
    train_iter, test_iter = IMDB()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * split_ratio)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    # tokenise the input samples
    tokenizer = get_tokenizer('basic_english')
    train_iter = IMDB(split='train')

    # build the vocubulary from the training set
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), min_freq = min_freq, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<pad>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 1 if x == 'pos' else 0

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch):
    label_list, text_list= [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    return label_list.to(device), text_list.to(device)

def word_embeddings(GloVe_name, GloVe_dim):
    # download the pre-trained word embeddings
    try:
        glove = torchtext.vocab.GloVe(name=GloVe_name, dim=GloVe_dim)
    except:
        glove = torchtext.vocab.GloVe(name='6B', dim=100)

    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, GloVe_dim))
    words_found = 0

    for i in range(matrix_len):
        try:
            word = vocab.lookup_token(i)
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(GloVe_dim, ))

    weights_matrix = torch.tensor(weights_matrix)
    glove_offset = -1 * torch.min(weights_matrix)
    glove_max = torch.max(weights_matrix) + glove_offset
    glove_offset, glove_max
    glove_matrix = (weights_matrix + glove_offset) / glove_max

    return glove_matrix
