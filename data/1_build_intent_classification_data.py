import os
import json
import argparse

import pandas as pd

from data.data_prepare_utils import prepare_dir

def get_intent_sentence(semantic_frame):
    intent = semantic_frame["intent"].rstrip()
    return intent

def build_data(fns, mode, domain):
    in_fn = fns["input"][mode]

    texts = []  # ~~~~ text
    intents = []  # intent text

    with open(in_fn, 'r', encoding='utf-8') as f:
        nlu_data = json.load(f)

        for idx, ex in enumerate(nlu_data):
            if domain == 'navi':
                ex = ex['positive']

            ## error check
            text = ex['text'].lstrip().rstrip()
            intent = ex['semantic_frame']['intent'].lstrip().rstrip()

            if text == '' or text == ' ' or intent == '' or intent == ' ' :
                print("Skipped due to error form, ", idx)
                continue

            # text
            text = ex['text'].lstrip().rstrip()
            text = text.replace('\n', '')  # delete \n in the middle fo the sentence
            texts.append(text)

            # intent text
            intent_text = get_intent_sentence(ex['semantic_frame']).lstrip().rstrip()
            intents.append(intent_text)

    ## check sanity
    assert (len(texts) == len(intents)), "Check data length"

    data = {
        "text" : texts,
        "intent_text" : intents
    }

    df = pd.DataFrame(data)

    # to dump
    to_fn = fns['output'][mode]
    df.to_csv(to_fn, sep='\t')
    print("[{}] Intent Classification Dataset File is dumped at".format(mode), to_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--domain", help="weather, navi", default="weather")
    args = parser.parse_args()

    domain = args.domain

    in_data_folder = os.path.join("./", domain)
    to_data_folder = os.path.join("./", domain, "run")
    prepare_dir(to_data_folder)

    fns = {
        "input" : {
            "train" : os.path.join(in_data_folder, "train.nlu.json"),
            "dev" : os.path.join(in_data_folder, "dev.nlu.json"),
            "test" : os.path.join(in_data_folder, "test.nlu.json")
        },
        "output" : {
            "train": os.path.join(to_data_folder, "train.nlu.tsv"),  # \t separator
            "dev" : os.path.join(to_data_folder, "dev.nlu.tsv"),
            "test" : os.path.join(to_data_folder, "test.nlu.tsv")
        }
    }

    if domain == "navi":
        fns["input"]["dev"] = os.path.join(in_data_folder, "test.nlu.json")

    build_data(fns, 'train', domain)
    build_data(fns, 'dev', domain)
    build_data(fns, 'test', domain)