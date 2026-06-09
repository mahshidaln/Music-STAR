"""Triplet network for metric learning.
Gently modified from 2020 InterDigital R&D and Télécom Paris. Authors: Ondřej Cífka, Brian McFee, Jongpil Lee, Juhan Nam
Gently modified from the ISMIR 2020 Tutorial for Metric Learning in MIR by Brian McFee, Jongpil Lee
and Juhan Nam, originally available at https://github.com/bmcfee/ismir2020-metric-learning/
and dedicated to the public domain under the CC0-1.0 license.
"""

import os

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    GlobalAvgPool1D,
    Input,
    Lambda,
    MaxPool1D,
    dot,
)
from tensorflow.keras.models import Model
from tqdm import tqdm


def build_model(num_features):
    """Build the triplet and backbone timbre models.

    Parameters
    ----------
    num_features : int
        Number of input MFCC features per frame.

    Returns
    -------
    tuple[tensorflow.keras.Model, tensorflow.keras.Model]
        Triplet model and shared backbone model.
    """

    def basic_block(x, num_features, fp_length):
        """Build one convolutional pooling block.

        Parameters
        ----------
        x
            Input Keras tensor.
        num_features : int
            Number of convolution filters.
        fp_length : int
            Filter and pooling length.

        Returns
        -------
        tensorflow.Tensor
            Output tensor.
        """

        x = Conv1D(
            num_features,
            fp_length,
            padding="same",
            use_bias=True,
            kernel_initializer="he_uniform",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool1D(pool_size=fp_length, padding="valid")(x)
        return x

    # Backbone model.
    num_frames = None
    x_in = Input(shape=(num_frames, num_features))
    x = basic_block(x_in, 64, 4)
    x = basic_block(x, 64, 4)
    x = basic_block(x, 64, 4)
    x = basic_block(x, 64, 2)
    x = GlobalAvgPool1D()(x)
    backbone_model = Model(inputs=[x_in], outputs=[x], name="backbone")
    backbone_model.summary()

    # Triplet model.
    anchor = Input(shape=(num_frames, num_features), name="anchor_input")
    positive = Input(shape=(num_frames, num_features), name="positive_input")
    negative = Input(shape=(num_frames, num_features), name="negative_input")

    anchor_embedding = backbone_model(anchor)
    positive_embedding = backbone_model(positive)
    negative_embedding = backbone_model(negative)

    # Cosine similarity.
    dist_fn = Lambda(lambda x: dot(x, axes=1, normalize=True))
    dist_anchor_positive = dist_fn([anchor_embedding, positive_embedding])
    dist_anchor_negative = dist_fn([anchor_embedding, negative_embedding])

    # Stack the similarity scores [1,0] and triplet model.
    similarity_scores = Lambda(lambda vects: K.stack(vects, axis=1))(
        [dist_anchor_positive, dist_anchor_negative]
    )
    tripletmodel = Model(
        inputs=[anchor, positive, negative], outputs=similarity_scores, name="triplet"
    )
    tripletmodel.summary()

    return tripletmodel, backbone_model


def triplet_hinge_loss(y_true, y_pred):
    """Compute triplet hinge loss.

    Parameters
    ----------
    y_true
        Unused target tensor.
    y_pred
        Similarity tensor whose first column is positive similarity and second
        column is negative similarity.

    Returns
    -------
    tensorflow.Tensor
        Scalar loss tensor.
    """
    del y_true
    # Always the first dimension of the similarity score is true.
    # Margin is set to 0.1
    y_pos = y_pred[:, 0]
    y_neg = y_pred[:, 1]
    loss = K.mean(K.maximum(0.0, 0.1 + y_neg - y_pos))
    return loss


def batch_triplet_loader(triplets, load_fn):
    """Load a batch of triplet examples.

    Parameters
    ----------
    triplets : Sequence[tuple[str, str, str]]
        Anchor, positive, and negative paths.
    load_fn : callable
        Function that loads one path into a feature array.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], numpy.ndarray]
        Keras input dictionary and target array.
    """

    anchor_col = []
    positive_col = []
    negative_col = []
    for p_anchor, p_positive, p_negative in triplets:
        a_anchor, a_positive, a_negative = (load_fn(p) for p in [p_anchor, p_positive, p_negative])
        # Stack batch data.

        anchor_col.append(a_anchor)
        positive_col.append(a_positive)
        negative_col.append(a_negative)

    # To array.
    anchor_col = np.array(anchor_col)
    positive_col = np.array(positive_col)
    negative_col = np.array(negative_col)

    batch_x = {
        "anchor_input": anchor_col,
        "positive_input": positive_col,
        "negative_input": negative_col,
    }

    batch_y = np.zeros((anchor_col.shape[0], 2))
    batch_y[:, 0] = 1
    return batch_x, batch_y


def data_loader(path, load_fn, batch_size, shuffle=False, repeat=False):
    """Create a generator over triplet batches.

    Parameters
    ----------
    path : str
        Triplet list file.
    load_fn : callable
        Function that loads one feature path.
    batch_size : int
        Batch size.
    shuffle : bool, optional
        Whether to shuffle triplets between epochs.
    repeat : bool, optional
        Whether to repeat forever.

    Returns
    -------
    tuple[callable, int]
        Loader factory and steps per epoch.
    """
    rng = np.random.default_rng(seed=0)

    base_path = os.path.dirname(path)
    with open(path) as f:
        dataset = [
            tuple(os.path.join(base_path, p) for p in line.rstrip("\n").split("\t")) for line in f
        ]

    steps_per_epoch = len(dataset) // batch_size

    def loader():
        """Yield triplet batches.

        Yields
        ------
        tuple[dict[str, numpy.ndarray], numpy.ndarray]
            Keras input dictionary and target array.
        """

        count_triplet = 0
        while True:
            if shuffle:
                rng.shuffle(dataset)

            for _ in range(0, steps_per_epoch * batch_size, batch_size):
                if count_triplet > len(dataset) - batch_size:
                    count_triplet = 0

                batch_x, batch_y = batch_triplet_loader(
                    dataset[count_triplet : count_triplet + batch_size], load_fn=load_fn
                )

                count_triplet += batch_size
                yield batch_x, batch_y

            if not repeat:
                break

    return loader, steps_per_epoch


def all_data(path, load_fn, shuffle=False):
    """Load all triplets into segmented arrays.

    Parameters
    ----------
    path : str
        Triplet list file.
    load_fn : callable
        Function that loads one feature path.
    shuffle : bool, optional
        Unused compatibility argument.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Anchor, positive, and negative segment arrays.
    """

    del shuffle
    anchor_col = []
    positive_col = []
    negative_col = []

    base_path = os.path.dirname(path)
    with open(path) as f:
        dataset = [
            tuple(os.path.join(base_path, p) for p in line.rstrip("\n").split("\t")) for line in f
        ]
    for triplet in tqdm(dataset):
        a_anchor, a_positive, a_negative = (
            load_fn(triplet[0]),
            load_fn(triplet[1]),
            load_fn(triplet[2]),
        )

        frames = int(16000 / 500 * 8)
        segments = int(len(a_anchor) // frames)
        for seg in range(segments):
            start = seg * frames
            end = (seg + 1) * frames
            anchor_col.append(a_anchor[start:end])
            positive_col.append(a_positive[start:end])
            negative_col.append(a_negative[start:end])

    return np.array(anchor_col), np.array(positive_col), np.array(negative_col)
