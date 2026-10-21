#!/usr/bin/env python3

import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "1",
)

import numpy as np
import tensorflow as tf


EXPERIMENT = "cloud-hpc-synthetic-heavy-v1"

SEED = 20260813

TRAIN_SAMPLES = 400_000
TEST_SAMPLES = 50_000

FEATURES = 256
CLASSES = 10

EPOCHS = 6
BATCH_SIZE = 2048

CPU_THREADS = int(
    os.environ.get(
        "CPU_THREADS",
        "16",
    )
)

RESULT_PATH = Path(
    os.environ.get(
        "RESULT_PATH",
        "/results/result_heavy.json",
    )
)


def max_rss_mb():
    value = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    # Linux reports ru_maxrss in KiB.
    return value / 1024.0


def make_dataset(
    rng,
    centers,
    samples,
):
    labels = rng.integers(
        low=0,
        high=CLASSES,
        size=samples,
        dtype=np.int32,
    )

    noise = rng.standard_normal(
        size=(
            samples,
            FEATURES,
        ),
        dtype=np.float32,
    )

    noise *= np.float32(0.70)

    features = (
        centers[labels]
        + noise
    ).astype(
        np.float32,
        copy=False,
    )

    return features, labels


def main():
    print(
        "====================================="
    )
    print(
        "Cloud-HPC Synthetic Heavy AI v1"
    )
    print(
        "====================================="
    )

    print(
        f"tensorflow_version={tf.__version__}"
    )
    print(
        f"numpy_version={np.__version__}"
    )

    print(f"seed={SEED}")
    print(
        f"train_samples={TRAIN_SAMPLES}"
    )
    print(
        f"test_samples={TEST_SAMPLES}"
    )
    print(f"features={FEATURES}")
    print(f"classes={CLASSES}")
    print(f"epochs={EPOCHS}")
    print(
        f"batch_size={BATCH_SIZE}"
    )
    print(
        f"cpu_threads={CPU_THREADS}"
    )

    np.random.seed(SEED)

    tf.keras.utils.set_random_seed(
        SEED
    )

    tf.config.threading.set_intra_op_parallelism_threads(
        CPU_THREADS
    )

    tf.config.threading.set_inter_op_parallelism_threads(
        2
    )

    rng = np.random.default_rng(
        SEED
    )

    data_start = time.perf_counter()

    centers = rng.standard_normal(
        size=(
            CLASSES,
            FEATURES,
        ),
        dtype=np.float32,
    )

    centers *= np.float32(4.0)

    x_train, y_train = make_dataset(
        rng,
        centers,
        TRAIN_SAMPLES,
    )

    x_test, y_test = make_dataset(
        rng,
        centers,
        TEST_SAMPLES,
    )

    data_seconds = (
        time.perf_counter()
        - data_start
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(FEATURES,)
            ),

            tf.keras.layers.Dense(
                512,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                256,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                CLASSES,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=[
            "accuracy"
        ],
    )

    train_start = (
        time.perf_counter()
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        shuffle=True,
    )

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    eval_start = (
        time.perf_counter()
    )

    test_loss, test_accuracy = (
        model.evaluate(
            x_test,
            y_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
    )

    eval_seconds = (
        time.perf_counter()
        - eval_start
    )

    # Warm-up before timing inference.
    model.predict(
        x_test[:BATCH_SIZE],
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    inference_start = (
        time.perf_counter()
    )

    model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    throughput = (
        TEST_SAMPLES
        / inference_seconds
    )

    result = {
        "experiment":
            EXPERIMENT,

        "seed":
            SEED,

        "train_samples":
            TRAIN_SAMPLES,

        "test_samples":
            TEST_SAMPLES,

        "features":
            FEATURES,

        "classes":
            CLASSES,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "cpu_threads":
            CPU_THREADS,

        "tensorflow_version":
            tf.__version__,

        "numpy_version":
            np.__version__,

        "data_seconds":
            round(
                data_seconds,
                6,
            ),

        "train_seconds":
            round(
                train_seconds,
                6,
            ),

        "eval_seconds":
            round(
                eval_seconds,
                6,
            ),

        "inference_seconds":
            round(
                inference_seconds,
                6,
            ),

        "inference_samples_per_second":
            round(
                throughput,
                3,
            ),

        "final_train_accuracy":
            round(
                float(
                    history.history[
                        "accuracy"
                    ][-1]
                ),
                6,
            ),

        "test_accuracy":
            round(
                float(
                    test_accuracy
                ),
                6,
            ),

        "test_loss":
            round(
                float(
                    test_loss
                ),
                6,
            ),

        "max_rss_mb":
            round(
                max_rss_mb(),
                3,
            ),
    }

    print(
        "====================================="
    )
    print("RESULT")
    print(
        "====================================="
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            result,
            fh,
            indent=2,
            sort_keys=True,
        )

        fh.write("\n")

    if test_accuracy < 0.95:
        print(
            "HEAVY_AI_BASELINE=FAIL"
        )

        return 2

    print(
        "HEAVY_AI_BASELINE=PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
