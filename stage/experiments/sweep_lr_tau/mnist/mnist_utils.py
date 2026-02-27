import os
import csv

RESULT_DIR = "./results"
RESULT_FILE = os.path.join(RESULT_DIR, "mnist_sgd_results.csv")


def init_results_file():
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "optimizer",
                "lr",
                "tau",
                "seed",
                "epoch",
                "epoch_loss",
                "epoch_acc",
                "epoch_max_grad_norm",
                "diverged",
            ])


def append_result(row_dict):
    with open(RESULT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row_dict["optimizer"],
            row_dict["lr"],
            row_dict["tau"],
            row_dict["seed"],
            row_dict["epoch"],
            row_dict["epoch_loss"],
            row_dict["epoch_acc"],
            row_dict["epoch_max_grad_norm"],
            row_dict["diverged"],
        ])