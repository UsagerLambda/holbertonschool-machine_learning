#!/usr/bin/env python3
# ruff: noqa: E402
"""Optimisation bayésienne d'hyperparamètres pour CNN CIFAR-10."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings

warnings.filterwarnings("ignore")

import time

import GPyOpt
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


keras = tf.keras
datasets = keras.datasets
layers = keras.layers
models = keras.models
regularizers = keras.regularizers


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class Optimizer:
    def __init__(self, max_iter=25, epochs=50):
        self.max_iter = max_iter
        self.epochs = epochs
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.optimizer = None

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = (
            datasets.cifar10.load_data()
        )
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        train_labels = keras.utils.to_categorical(train_labels, 10)
        test_labels = keras.utils.to_categorical(test_labels, 10)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )

        return test_images, test_labels


    def build_model(self, dense_units, dropout, l2_reg, hidden_layers, learning_rate):
        """Construit le modèle CNN"""
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.Dropout(0.2),
                layers.Flatten(),
            ]
        )

        for _ in range(hidden_layers):
            model.add(
                layers.Dense(
                    dense_units,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_reg),
                )
            )
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(10))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        return model


    def objective_function(self, hyperparams):
        """Fonction objectif à minimiser"""
        start_time = time.time()

        batch_size = int(hyperparams[0, 0])
        learning_rate = float(hyperparams[0, 1])
        dropout = float(hyperparams[0, 2])
        dense_units = int(hyperparams[0, 3])
        l2_reg = float(hyperparams[0, 4])
        hidden_layers = int(hyperparams[0, 5])

        print(f"\n{Colors.CYAN}{Colors.BOLD}[TESTING]{Colors.RESET}")
        print(f"  {Colors.BLUE}Batch size:{Colors.RESET}      {batch_size}")
        print(f"  {Colors.BLUE}Learning rate:{Colors.RESET}   {learning_rate:.5f}")
        print(f"  {Colors.BLUE}Dropout:{Colors.RESET}         {dropout:.2f}")
        print(f"  {Colors.BLUE}Dense units:{Colors.RESET}     {dense_units}")
        print(f"  {Colors.BLUE}L2 reg:{Colors.RESET}          {l2_reg:.5f}")
        print(f"  {Colors.BLUE}Hidden layers:{Colors.RESET}   {hidden_layers}")

        model = self.build_model(
            dense_units, dropout, l2_reg, hidden_layers, learning_rate
        )

        filename = f"models/model_bs{batch_size}_lr{learning_rate:.5f}_drop{dropout:.2f}_units{dense_units}_l2{l2_reg:.5f}_layers{hidden_layers}.keras"

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=filename,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )

        model.fit(
            self.X_train,
            self.Y_train,
            batch_size=batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, self.Y_val),
            callbacks=[checkpoint, early_stop],
            verbose=0,
        )

        Y_pred = model.predict(self.X_val, verbose=0)
        Y_pred_classes = tf.argmax(Y_pred, axis=1).numpy()
        Y_val_classes = tf.argmax(self.Y_val, axis=1).numpy()

        f1 = f1_score(Y_val_classes, Y_pred_classes, average="macro")

        elapsed_time = time.time() - start_time

        print(f"  {Colors.GREEN}F1-score:{Colors.RESET}         {f1:.4f}")
        print(f"  {Colors.YELLOW}Time:{Colors.RESET}            {elapsed_time:.1f}s")

        return 1 - f1


    def run(self):
        """Lance l'optimisation bayésienne"""
        bounds = [
            {"name": "batch_size", "type": "discrete", "domain": (32, 64, 128)},
            {"name": "learning_rate", "type": "continuous", "domain": (0.0001, 0.003)},
            {"name": "dropout", "type": "continuous", "domain": (0.1, 0.4)},
            {"name": "dense_units", "type": "discrete", "domain": (128, 256)},
            {"name": "l2_reg", "type": "continuous", "domain": (0.0001, 0.005)},
            {"name": "hidden_layers", "type": "discrete", "domain": (1, 2)},
        ]

        self.optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.objective_function,
            domain=bounds,
            model_type="GP",
            acquisition_type="EI",
            acquisition_jitter=0.05,
            exact_feval=True,
            maximize=False,
        )

        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.MAGENTA}BAYESIAN OPTIMIZATION - STARTING{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'=' * 60}{Colors.RESET}\n")

        total_start = time.time()

        self.optimizer.run_optimization(max_iter=self.max_iter)

        total_time = time.time() - total_start

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}OPTIMIZATION COMPLETED{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 60}{Colors.RESET}")
        print(
            f"{Colors.BRIGHT_WHITE}Total time: {Colors.BOLD}{total_time / 60:.1f}{Colors.RESET}{Colors.BRIGHT_WHITE} minutes{Colors.RESET}\n"
        )

        return total_time


    def save_results(self, total_time):
        """Sauvegarde les résultats de l'optimisation"""
        if self.optimizer is None:
            raise RuntimeError(
                "L'optimisation doit être lancée avant de sauvegarder les résultats"
            )

        opt = self.optimizer

        os.makedirs("img", exist_ok=True)
        opt.plot_convergence()
        plt.savefig("img/convergence_plot.png", dpi=150, bbox_inches="tight")
        print(
            f"{Colors.CYAN}[SAVED]{Colors.RESET} Plot saved to: {Colors.DIM}img/convergence_plot.png{Colors.RESET}"
        )

        best_params = opt.x_opt
        best_f1 = 1 - opt.fx_opt

        with open("bayes_opt.txt", "w") as f:
            f.write("=" * 60 + "\n")
            f.write("BAYESIAN OPTIMIZATION REPORT - CIFAR-10 CNN\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total optimization time: {total_time / 60:.2f} minutes\n")
            f.write(f"Number of iterations: {len(opt.Y)}\n\n")

            f.write("BEST HYPERPARAMETERS FOUND:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Batch size:         {int(best_params[0])}\n")
            f.write(f"  Learning rate:      {best_params[1]:.6f}\n")
            f.write(f"  Dropout:            {best_params[2]:.4f}\n")
            f.write(f"  Dense units:        {int(best_params[3])}\n")
            f.write(f"  L2 regularization:  {best_params[4]:.6f}\n")
            f.write(f"  Hidden layers:      {int(best_params[5])}\n\n")

            f.write(f"BEST F1-SCORE: {best_f1:.4f}\n\n")

            f.write("ALL EVALUATIONS:\n")
            f.write("=" * 60 + "\n")
            for i, (X, Y) in enumerate(zip(opt.X, opt.Y)):
                f1_iter = 1 - Y[0]
                f.write(f"Iter {i + 1:2d} | F1: {f1_iter:.4f} | ")
                f.write(f"bs={int(X[0]):3d} lr={X[1]:.5f} drop={X[2]:.2f} ")
                f.write(f"units={int(X[3]):3d} l2={X[4]:.5f} layers={int(X[5])}\n")

        print(
            f"{Colors.CYAN}[SAVED]{Colors.RESET} Results saved to: {Colors.DIM}bayes_opt.txt{Colors.RESET}"
        )


def main():
    """Point d'entrée principal"""
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}CIFAR-10 HYPERPARAMETER OPTIMIZATION{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'=' * 60}{Colors.RESET}\n")

    optimizer = Optimizer(max_iter=25, epochs=50)

    print(f"{Colors.YELLOW}[LOADING]{Colors.RESET} Loading CIFAR-10 dataset...")
    X_test, Y_test = optimizer.load_data()

    print(f"\n{Colors.BOLD}Dataset shapes:{Colors.RESET}")
    print(f"  {Colors.BLUE}Training:{Colors.RESET}   {optimizer.X_train.shape}")
    print(f"  {Colors.BLUE}Validation:{Colors.RESET} {optimizer.X_val.shape}")
    print(f"  {Colors.BLUE}Test:{Colors.RESET}       {X_test.shape}")

    total_time = optimizer.run()

    optimizer.save_results(total_time)


if __name__ == "__main__":
    main()
