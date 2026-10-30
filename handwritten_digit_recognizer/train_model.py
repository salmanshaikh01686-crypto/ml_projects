from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale pixel intensities to [0, 1] and add the grayscale channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Training images: {x_train.shape}; test images: {x_test.shape}")
    model = build_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_mnist_cnn.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
    ]

    print("Training CNN. First run downloads MNIST and may take several minutes...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    best_path = MODEL_DIR / "best_mnist_cnn.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)

    print("Evaluating on the untouched MNIST test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    final_model = MODEL_DIR / "mnist_cnn.keras"
    model.save(final_model)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(MODEL_DIR / "training_history.csv", index=False)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": int(len(history_df)),
        "dataset": "MNIST",
        "architecture": "3 convolution blocks + dense classifier",
        "seed": SEED,
    }
    (MODEL_DIR / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Saved model: {final_model}")
    print("Now run: py -m streamlit run app.py")

if __name__ == "__main__":
    main()
