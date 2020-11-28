import os
import argparse

import numpy as np
import pandas as pd

def prepare_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def load_vocab(fn):
    print("Vocab loading from {}".format(fn))

    vocab = {}
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            symbol, _id = line.split('\t')
            vocab[symbol] = int(_id)

    return vocab

def make_label_dir(image_dir, vocab):
    for v in vocab.keys():
        if v == '<PAD>':
            continue
        label_dir = os.path.join(image_dir, v)
        prepare_dir(label_dir)

def convert_vector2graph(fns, rep_type, top_n, need_edges, mode):
    v_t_fn = fns["input"]["v_t"][mode]
    text_df_fn = fns["input"]["text"][mode]
    label_vocab_fn = fns["input"]["text"]["label_vocabs"]

    output_fn = fns["output"][mode]

    # loading vector movement
    vector_movement = np.load(v_t_fn)
    print("[{}]Vector Movement file loading at ".format(mode), v_t_fn, " - {}".format(vector_movement.shape))

    # getting labels
    text_df = pd.read_csv(text_df_fn, sep='\t')
    labels = text_df["intent_text"].tolist()

    # prepare directories for using ImageFolder of Torchvision
    label_vocab = load_vocab(label_vocab_fn) # label name is folder name at ImageFolder
    make_label_dir(output_fn, label_vocab)

    # converting vector 2 graph
    if rep_type == "flatted_bar":
        raise NotImplementedError("Flatted Bar is not supported in this version.")
    elif rep_type == "circular_bar":
        raise NotImplementedError("Circular Bar is not supported in this version.")
    elif rep_type == "graph":
        from vector2graph.representation.graph_converter import build_graph
        build_graph(labels, vector_movement, output_fn, need_edges=need_edges, top_n=top_n)
    else:
        raise KeyError(rep_type)

if __name__ == '__main__':
    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # domain -------------------------------------------------------------------------------------
    parser.add_argument("--domain", default="weather",
                        help="What domain do you want?")

    # text reader --------------------------------------------------------------------------------
    parser.add_argument("--text_reader", default="bert",
                        help="When generating vector movement, What text reader was used ?") # fixed to bert, now

    # params for generated vector movements ------------------------------------------------------
    parser.add_argument("--num_samples", default=50,
                        help="number of dropout example for generating perturbation")

    # params for converting representation -------------------------------------------------------
    parser.add_argument("--rep_type", default="graph",
                        help="vector To ? --> graph, circular_bar, flatted_bar") # fixed to graph in this version
    parser.add_argument("--need_edges", action='store_true',
                        help="Do you wanna edges(links between nodes) at generating graph ?")
    parser.add_argument("--top_n", help='Select Number of Dimensions For building Representations',
                        default=10)

    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # specific domain --------------------------------------------------------------------------------------------------
    domain = args.domain
    # ------------------------------------------------------------------------------------------------------------------

    # fns --------------------------------------------------------------------------------------------------------------
    text_data_dir = os.path.join("data", domain, "run")

    text_reader = args.text_reader
    v_t_dir = os.path.join("model", domain, text_reader)
    num_samples = int(args.num_samples)

    save_dir = os.path.join("images", domain, text_reader)
    rep_type = args.rep_type
    need_edges = args.need_edges
    edge_flag = "with_edge" if need_edges else "wo_edge"
    top_n = int(args.top_n)

    fns = {
        "input" : {
            "text" : {
                "train" : os.path.join(text_data_dir, "train.nlu.tsv"),
                "test" : os.path.join(text_data_dir, "test.nlu.tsv"),
                "label_vocabs": os.path.join(text_data_dir, "intent.vocab"),
            },
            "v_t": {
                "train": os.path.join(v_t_dir, "train.v_t.{}.npy".format(num_samples)),
                "test": os.path.join(v_t_dir, "test.v_t.{}.npy".format(num_samples))
            }
        },
        "output" : {
            "train" : os.path.join(save_dir, "{}_{}".format(rep_type, num_samples), "top_{}_{}".format(top_n, edge_flag), "train"),
            "test" : os.path.join(save_dir, "{}_{}".format(rep_type, num_samples), "top_{}_{}".format(top_n, edge_flag), "test")
        }
    }
    # ------------------------------------------------------------------------------------------------------------------

    # Do convert vector 2 graph
    convert_vector2graph(fns, rep_type, top_n, need_edges, "train")
    convert_vector2graph(fns, rep_type, top_n, need_edges, "test")