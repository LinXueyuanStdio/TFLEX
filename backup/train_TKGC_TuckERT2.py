import argparse

from temp.load_data import Data
from temp.train import train_temporal
from toolbox.nn.TuckERCPD import TuckERCPD
from toolbox.nn.TuckERT import TuckERT
from toolbox.nn.TuckERTNT import TuckERTNT
from toolbox.nn.TuckERTTR import TuckERTTR
from toolbox.nn.TuckERTTT import TuckERTTT


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TuckERT", nargs="?", help="Which model to use: TuckERT,TucERTNT, TuckERTTR,TuckERCPD.")
    parser.add_argument("--dataset", type=str, default="ICEWS14", nargs="?", help="Which dataset to use: ICEWS14, ICEWS05-15.")
    parser.add_argument("--n_iter", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, nargs="?", help="Learning rate.")
    parser.add_argument("--de", type=int, default=50, nargs="?", help="Entity embedding dimensionality.")
    parser.add_argument("--dr", type=int, default=50, nargs="?", help="Relation embedding dimensionality.")
    parser.add_argument("--dt", type=int, default=50, nargs="?", help="Temporal embedding dimensionality.")
    parser.add_argument("--ranks", type=int, default=10, nargs="?", help="Ranks of tensor for TR/TT model.")
    parser.add_argument("--device", type=str, default="cuda", nargs="?", help="Device to run the code on. Either cuda or cpu")
    parser.add_argument("--early_stopping", type=int, default=False, nargs="?", help="Early stopping value")
    parser.add_argument("--input_dropout", type=float, default=0., nargs="?", help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0., nargs="?", help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0., nargs="?", help="Dropout after the second hidden layer.")
    parser.add_argument("--hidden_dropout3", type=float, default=0., nargs="?", help="Dropout after the third hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0., nargs="?", help="Amount of label smoothing.")

    args = parser.parse_args()

    return args


def train_model_from_args(args, print_scores=True):
    data_dir = "data/" + args.dataset + "/"
    data = Data(data_dir=data_dir)

    model_dict = {
        "TuckERCPD": TuckERCPD,
        "TuckERT": TuckERT,
        "TuckERTNT": TuckERTNT,
        "TuckERTTR": TuckERTTR,
        "TuckERTTT": TuckERTTT,
    }
    model = model_dict[args.model](d=data, **vars(args)).to(args.device)

    print("\n----------------------------------- TRAINING -------------------------")
    model, metrics = train_temporal(model, data, device=args.device, n_iter=args.n_iter, learning_rate=args.learning_rate, batch_size=args.batch_size, early_stopping=args.early_stopping)

    if print_scores:
        print("\n----------------------------------- Metrics --------------------------\n")
        print(metrics)

    return metrics


def main():
    args = parse()
    train_model_from_args(args, True)


if __name__ == '__main__':
    main()
