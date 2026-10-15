from __future__ import annotations
from typing import Any


def _tf():
    try:
        import tensorflow as tf
    except ImportError as e:
        raise RuntimeError(
            "TensorFlow is required for Phase 07. Install with: pip install tensorflow"
        ) from e
    return tf


def build_gru_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    *,
    units_1: int = 128,
    units_2: int = 64,
    dense_units: int = 64,
    dropout: float = 0.30,
    learning_rate: float = 1e-3,
):
    tf = _tf()

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="sequence",
    )

    x = tf.keras.layers.Masking(
        mask_value=0.0,
        name="padding_mask",
    )(inputs)

    x = tf.keras.layers.GRU(
        units_1,
        return_sequences=True,
        name="gru_128",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout,
        name="dropout_1",
    )(x)

    x = tf.keras.layers.GRU(
        units_2,
        return_sequences=False,
        name="gru_64",
    )(x)

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="dense_64",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout,
        name="dropout_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="sign_softmax",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="phase7_gru",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_tcn_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    *,
    filters: int = 128,
    kernel_size: int = 3,
    dilations=(1,2,4,8),
    dense_units: int = 64,
    dropout: float = 0.30,
    learning_rate: float = 1e-3,
):
    """
    Lightweight TCN comparison model.

    Padding rows are exact zero after preprocessing. Convolutions use no bias
    and ReLU, so padded zeros remain neutral before GlobalMaxPooling1D.
    """
    tf = _tf()

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="sequence",
    )

    x = inputs

    for i, dilation in enumerate(dilations):
        residual = x

        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(dilation),
            padding="causal",
            use_bias=False,
            name=f"tcn_conv_d{dilation}",
        )(x)

        x = tf.keras.layers.BatchNormalization(
            name=f"tcn_bn_d{dilation}"
        )(x)

        x = tf.keras.layers.ReLU(
            name=f"tcn_relu_d{dilation}"
        )(x)

        x = tf.keras.layers.Dropout(
            dropout,
            name=f"tcn_dropout_d{dilation}",
        )(x)

        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(
                filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                name=f"tcn_residual_projection_{i}",
            )(residual)

        x = tf.keras.layers.Add(
            name=f"tcn_residual_add_{i}"
        )([x, residual])

        x = tf.keras.layers.ReLU(
            name=f"tcn_residual_relu_{i}"
        )(x)

    x = tf.keras.layers.GlobalMaxPooling1D(
        name="temporal_global_max_pool"
    )(x)

    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="dense_64",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout,
        name="final_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="sign_softmax",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="phase7_tcn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_model(
    model_name: str,
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    config: dict[str, Any],
):
    training = config.get("training", {}) or {}
    lr = float(training.get("learning_rate", 1e-3))

    name = model_name.lower()

    if name == "gru":
        m = config.get("gru", {}) or {}
        return build_gru_model(
            sequence_length,
            feature_dim,
            num_classes,
            units_1=int(m.get("units_1", 128)),
            units_2=int(m.get("units_2", 64)),
            dense_units=int(m.get("dense_units", 64)),
            dropout=float(m.get("dropout", 0.30)),
            learning_rate=lr,
        )

    if name == "tcn":
        m = config.get("tcn", {}) or {}
        return build_tcn_model(
            sequence_length,
            feature_dim,
            num_classes,
            filters=int(m.get("filters", 128)),
            kernel_size=int(m.get("kernel_size", 3)),
            dilations=tuple(m.get("dilations", [1,2,4,8])),
            dense_units=int(m.get("dense_units", 64)),
            dropout=float(m.get("dropout", 0.30)),
            learning_rate=lr,
        )

    raise ValueError("model_name must be 'gru' or 'tcn'")
