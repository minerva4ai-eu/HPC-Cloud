import os
import json
import time
import resource
from pathlib import Path

# CPU-only baseline
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf


SEED = 20260813
N_TRAIN = 120000
N_TEST = 20000
N_FEATURES = 64
N_CLASSES = 10
EPOCHS = 8
BATCH_SIZE = 512
CPU_THREADS = 4


# --------------------------------------
# Reproducibility / CPU configuration
# --------------------------------------

tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.keras.utils.set_random_seed(SEED)

print("=====================================")
print("Cloud-HPC CPU AI Baseline v1")
print("=====================================")

print(f"tensorflow_version={tf.__version__}")
print(f"numpy_version={np.__version__}")
print(f"seed={SEED}")
print(f"train_samples={N_TRAIN}")
print(f"test_samples={N_TEST}")
print(f"features={N_FEATURES}")
print(f"classes={N_CLASSES}")
print(f"epochs={EPOCHS}")
print(f"batch_size={BATCH_SIZE}")
print(f"cpu_threads={CPU_THREADS}")


# --------------------------------------
# Deterministic synthetic dataset
# --------------------------------------

data_start = time.perf_counter()

rng = np.random.default_rng(SEED)

prototypes = rng.normal(
    loc=0.0,
    scale=3.0,
    size=(N_CLASSES, N_FEATURES),
).astype(np.float32)


def make_dataset(n_samples):
    labels = rng.integers(
        low=0,
        high=N_CLASSES,
        size=n_samples,
        dtype=np.int32,
    )

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, N_FEATURES),
    ).astype(np.float32)

    features = prototypes[labels] + noise

    return features, labels


x_train, y_train = make_dataset(N_TRAIN)
x_test, y_test = make_dataset(N_TEST)

data_seconds = time.perf_counter() - data_start


# --------------------------------------
# MLP model
# --------------------------------------

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(N_FEATURES,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# --------------------------------------
# Training
# --------------------------------------

train_start = time.perf_counter()

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=2,
)

train_seconds = time.perf_counter() - train_start


# --------------------------------------
# Evaluation
# --------------------------------------

eval_start = time.perf_counter()

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=0,
)

eval_seconds = time.perf_counter() - eval_start


# --------------------------------------
# Inference throughput
# --------------------------------------

infer_start = time.perf_counter()

_ = model(x_test, training=False).numpy()

inference_seconds = time.perf_counter() - infer_start

samples_per_second = (
    N_TEST / inference_seconds
    if inference_seconds > 0
    else 0.0
)


# Linux ru_maxrss is reported in KiB
max_rss_mb = (
    resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024.0
)


result = {
    "experiment": "cloud-hpc-cpu-ai-v1",
    "tensorflow_version": tf.__version__,
    "numpy_version": np.__version__,
    "seed": SEED,
    "train_samples": N_TRAIN,
    "test_samples": N_TEST,
    "features": N_FEATURES,
    "classes": N_CLASSES,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "cpu_threads": CPU_THREADS,
    "data_seconds": round(data_seconds, 6),
    "train_seconds": round(train_seconds, 6),
    "eval_seconds": round(eval_seconds, 6),
    "inference_seconds": round(inference_seconds, 6),
    "inference_samples_per_second": round(
        samples_per_second, 3
    ),
    "final_train_accuracy": round(
        float(history.history["accuracy"][-1]),
        6,
    ),
    "test_loss": round(float(test_loss), 6),
    "test_accuracy": round(float(test_accuracy), 6),
    "max_rss_mb": round(max_rss_mb, 3),
}


Path("/results").mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    "/results/result.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
        sort_keys=True,
    )


print("=====================================")
print("RESULT")
print("=====================================")

print(json.dumps(result, indent=2, sort_keys=True))


# Simple correctness gate
if test_accuracy < 0.95:
    print("CPU_AI_BASELINE=FAIL")
    raise SystemExit(1)

print("CPU_AI_BASELINE=PASS")
