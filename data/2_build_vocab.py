import os
import argparse

def build_vocab(fns):
    import pandas as pd
    train_df = pd.read_csv(fns["input"]["train"], sep='\t')
    test_df = pd.read_csv(fns["input"]["test"], sep='\t')
    dev_df = pd.read_csv(fns["input"]["dev"], sep='\t')

    # intent coverage check
    _train_set = set(train_df['intent_text'].unique().tolist())
    _test_set = set(test_df['intent_text'].unique().tolist())
    _dev_set = set(dev_df['intent_text'].unique().tolist())

    # validation
    assert len(_test_set - _train_set) <= 0, "Intent tags in test set are not in train"
    assert len(_dev_set - _train_set) <= 0, "Intent tags in dev set are not in train"

    # building vocab
    intent_vocab = ['<PAD>'] + [x for x in list(sorted(list(_train_set)))]

    # dumping vocab file
    intent_vocab_fn = fns["output"]["intent"]
    with open(intent_vocab_fn, 'w', encoding='utf-8') as f:
        for idx, intent in enumerate(intent_vocab):
            print("{}\t{}".format(intent, idx), file=f)
        print("[Intent] vocab is dumped at ", intent_vocab_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--domain", help="weather, navi", default="weather")
    args = parser.parse_args()

    domain = args.domain
    data_folder = os.path.join("./", domain, "run")

    fns = {
        "input": {
            "train" : os.path.join(data_folder, "train.nlu.tsv"),
            "dev" : os.path.join(data_folder, "dev.nlu.tsv"),
            "test" : os.path.join(data_folder, "test.nlu.tsv")
        },
        "output": {
            "intent" : os.path.join(data_folder, "intent.vocab")
        }
    }

    if domain == "navi":
        fns["input"]["dev"] = os.path.join(data_folder, "test.nlu.tsv")

    build_vocab(fns)