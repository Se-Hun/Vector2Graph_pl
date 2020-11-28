import os
import argparse

from data.data_prepare_utils import prepare_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="weather, navi", default="weather")
    args = parser.parse_args()

    domain = args.domain

    run_dir = os.path.join("./", domain, "run")
    prepare_dir(run_dir)

    cmd = 'python 1_build_intent_classification_data.py --domain={}'.format(domain)
    print("Execute : ", cmd)
    os.system(cmd)

    cmd = 'python 2_build_vocab.py --domain={}'.format(domain)
    print("Execute : ", cmd)
    os.system(cmd)
