import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class Intent_Classification_Dataset(Dataset):
    def __init__(self, df, tokenizer, intent_label_vocab, max_seq_len):
        self.tokenizer = tokenizer
        self.intent_label_vocab = intent_label_vocab

        self.max_seq_len = max_seq_len

        # for debugging -- to smallset
        # N = 70
        # df = df[:N]

        # transform all data
        from tqdm.auto import tqdm
        df_iterator = tqdm(df.iterrows(), desc="Iteration")

        self.texts = []
        self.intents = []
        for row_idx, (index, row) in enumerate(df_iterator):
            text_objs = self.proc_text(str(row['text']), self.tokenizer)
            intent_id = self.proc_intent_text(row['intent_text'])

            self.texts.append(text_objs)
            self.intents.append(intent_id)

    def proc_text(self, text, tokenizer):
        obj = tokenizer(text, padding='max_length', max_length=self.max_seq_len, truncation=True)
        if 'token_type_ids' not in obj:
            obj['token_type_ids'] = [0] * len(obj['input_ids'])
        return obj

    def proc_intent_text(self, intent_text):
        intent_id = self.intent_label_vocab[intent_text]
        return intent_id

    def __getitem__(self, i):
        input_ids = np.array(self.texts[i]['input_ids'])
        token_type_ids = np.array(self.texts[i]['token_type_ids'])
        attention_mask = np.array(self.texts[i]['attention_mask'])

        intent_ids = np.array(self.intents[i])

        item = [input_ids, token_type_ids, attention_mask, intent_ids]
        return item

    def __len__(self):
        return (len(self.texts))

class Intent_Classification_Data_Module(pl.LightningDataModule):
    def __init__(self, domain, text_reader, max_seq_length, batch_size):
        super().__init__()

        # prepare tokenizer
        from utils import get_tokenizer
        self.tokenizer = get_tokenizer(domain, text_reader)

        # data preparing params
        self.data_dir = os.path.join("./data", domain, "run")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        # after preparing data, we will return number of intents.
        self.num_intents = None

    def prepare_data(self):
        # vocab
        intent_label_vocab = self._load_vocab(os.path.join(self.data_dir, "intent.vocab"))
        self.num_intents = len(intent_label_vocab)

        # read data
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.nlu.tsv"), sep='\t')
        valid_df = pd.read_csv(os.path.join(self.data_dir, "dev.nlu.tsv"), sep='\t')
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.nlu.tsv"), sep='\t')

        # building dataset
        self.train_dataset = Intent_Classification_Dataset(train_df, self.tokenizer, intent_label_vocab, self.max_seq_length)
        self.valid_dataset = Intent_Classification_Dataset(valid_df, self.tokenizer, intent_label_vocab, self.max_seq_length)
        self.test_dataset = Intent_Classification_Dataset(test_df, self.tokenizer, intent_label_vocab, self.max_seq_length)

    def _load_vocab(self, fn):
        print("Vocab loading from {}".format(fn))

        vocab = {}
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                symbol, _id = line.split('\t')
                vocab[symbol] = int(_id)

        return vocab

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def train_dataloader_for_dump(self):
        return DataLoader(self.train_dataset, batch_size=1) # fixed batch_size=1

    def test_dataloader_for_dump(self):
        return DataLoader(self.test_dataset, batch_size=1) # fixed batch_size=1
