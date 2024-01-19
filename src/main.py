import argparse
import os
import random
import signal
import sys
import traceback
import train.regular as regular
import numpy as np
import torch

import classifier.factory as clf
import dataset.loader as loader
from embedding.embedmodel import EMBED_BERT


def parse_args():
    parser = argparse.ArgumentParser(description="Label-Guided Distance Scaling for Few-Shot Text Classification")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="../data/huffpost.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="huffpost",
                        help="name of the dataset. "
                             "Options: [20newsgroup, amazon, huffpost, "
                             "reuters, rcv1, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=20,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=16,
                        help="number of meta-test classes")
    # extra arguments for domain configuration
    parser.add_argument("--n_train_domain", type=int, default=1,
                        help="number of meta-train domains")
    parser.add_argument("--n_val_domain", type=int, default=1,
                        help="number of meta-val domains")
    parser.add_argument("--n_test_domain", type=int, default=1,
                        help="number of meta-test domains")
    parser.add_argument("--seed", type=int, default=2023)
    # extra arguments for data process
    parser.add_argument("--max_text_len_limits", type=int, default=24,
                        help="max number of tokens for each sample")
    parser.add_argument("--cross_domain", action="store_true", default=False,
                        help=("whether train and test the cross-domain ability of the model"))

    # extra arguments for BERT
    parser.add_argument("--fixbert", type=str, default="no",
                        help="no: do not fix bert"
                             "ebd: fix bert ebd"
                             "encoder: fix bert encoder"
                             "all: fix all parameters in bert")

    # load bert embeddings for sent-level datasets (optional)
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Num. of cores used for loading data. Set this "
                             "to zero if you want to use all the cpus.")
    parser.add_argument("--bert", default=True, action="store_true",
                        help=("set true if use bert embeddings "
                              "(only available for sent-level datasets: "
                              "huffpost, fewrel"))
    parser.add_argument("--bert_cache_dir", default='../src/~/.pytorch_pretrained_bert/', type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default='bert-base-uncased', type=str,
                        help=("path to the pre-trained bert embeddings."))

    # task configuration
    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    # train/test configuration
    # 20, 10, 20, 1000 for tars
    parser.add_argument("--train_epochs", type=int, default=50,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    # model options
    parser.add_argument("--embedding", type=str, default="ebdnew",
                        help=("document embedding method. Options: "
                              "[avg, tfidf, meta, oracle, cnn, ebdnew]"))
    parser.add_argument("--classifier", type=str, default="mbc",
                        help=("classifier. Options: [nn, proto, r2d2, mlp]"))
    parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                        help=("auxiliary embeddings (used for fewrel). "
                              "Options: [pos, ent]"))
    parser.add_argument("--sim", type=str, default="cos",
                        help="l2: use l2 distance"
                             "cos: use cosine distance")

    parser.add_argument("--cuda", type=int, default=0,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="train",
                        help=("Running mode."
                              "Options: [train, test, finetune]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=True,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--patience", type=int, default=10, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    # save
    parser.add_argument("--result_path", type=str,
                        default="result/{}_result.csv")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")
    parser.add_argument("--layer_number", type=int, default=12)
    parser.add_argument("--all_layers", action="store_true", default=False)
    parser.add_argument("--all_layers_numbera", type=int, default=1)
    parser.add_argument("--all_layers_numberb", type=int, default=12)
    parser.add_argument("--add_mlp", action="store_true", default=False)
    parser.add_argument("--template", type=str,
                        default="[MASK] [sentence]", )
    parser.add_argument("--pool", type=str,
                        default='cls')
    parser.add_argument("--T", type=float, default=1)
    # baseline
    parser.add_argument("--add_instance", action="store_true",  default=False)
    parser.add_argument("--add_pro", action="store_true", default=False)
    parser.add_argument("--alpha_pro", type=float, default=1)
    # baseline2
    parser.add_argument("--add_cos", action="store_true", default=False)
    # ours
    parser.add_argument("--add_prol", action="store_true",  default=False)
    parser.add_argument("--add_prosq", action="store_true",  default=False)
    parser.add_argument("--alpha_pl", type=float, default=1)
    parser.add_argument("--alpha_prosq", type=float, default=1)

    parser.add_argument("--protype", type=str,  default="single")
    parser.add_argument("--cltype", type=str, default="proto")
    parser.add_argument("--SG", type=str, default='mean')
    parser.add_argument("--falpha", type=float, default=0.1,
                        help='falpha * label + (1 - falpha) * sentence')
    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))



def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    import warnings
    warnings.filterwarnings('ignore')
    args = parse_args()
    print_args(args)
    set_seed(args.seed)

    # load data

    train_data, val_data, test_data = loader.load_data(args)

    # initialize model
    model = {}

    # model["ebd"] = MY_LASMAL(args)
    model["ebd"] = EMBED_BERT(args)

    args.device = torch.device("cuda:{}".format(args.cuda)
                               if args.cuda != -1 else "cpu")

    model['ebd'].to(args.device)
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)
    model["clf"].to(args.device)

    # import pdb
    # pdb.set_trace()
    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        regular.train(train_data, val_data, model, args)

    # In finetune, we combine all train and val classes and split it into train
    # and validation examples.
    if args.mode != "finetune":
        print("start test by val data")
        state = "val"
        if args.classifier == 'mbcnew':
            val_acc, val_std, val_acc_knn, val_std_knn, val_acc_proen, val_std_proen = regular.test(val_data, model, args,
                                                                                                    args.val_episodes, state=state)
        else:
            val_acc, val_std = regular.test(val_data, model, args,
                                            args.val_episodes, state=state)
    else:
        val_acc, val_std = 0, 0

    # args.mode = 'test'
    if args.classifier == 'mbcnew':
        test_acc, test_std, test_acc_knn, test_std_knn, test_acc_proen, test_std_proen = regular.test(test_data, model, args,
                                                                                                      args.test_episodes, state="test")
    else:
        test_acc, test_std = regular.test(test_data, model, args,
                                          args.test_episodes, state="test")

    if args.result_path:
        args.result_path = args.result_path.format(
            str(args.way) + "-way_" + str(args.shot) + "-shot_" + args.dataset)
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdir(directory)

        if args.classifier == 'mbcnew':
            result = {
                "test_acc": test_acc,
                "test_std": test_std,
                "val_acc": val_acc,
                "val_std": val_std,
                "test_acc_knn": test_acc_knn,
                "test_std_knn": test_std_knn,
                "test_acc_proen": test_acc_proen,
                "test_std_proen": test_std_proen,
            }
        else:
            result = {
                "test_acc": test_acc,
                "test_std": test_std,
                "val_acc": val_acc,
                "val_std": val_std,
            }

        import csv
        colmuns = []
        data = []
        with open(args.result_path, "a", newline="") as f:
            for key in result.keys():
                colmuns.append(key)
                data.append(result[key])
            for attr, value in sorted(args.__dict__.items()):
                colmuns.append(attr.upper())
                data.append(value)
            writer = csv.writer(f)
            # writer.writerow(colmuns)
            writer.writerow(data)


if __name__ == "__main__":
    try:
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        main()
    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        os.killpg(0, signal.SIGKILL)

    exit(0)
