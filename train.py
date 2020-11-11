"""The training script for BlazeFace"""
import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from callbacks import LogImages
from network import blaze_net
from dataset import build_dataset_from_wider
from losses import BlazeLoss

parser = ArgumentParser()
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
parser.add_argument("--eval_only", default=False, type=bool,
                    help="Evaluate the model without training.")
args = parser.parse_args()


if __name__ == "__main__":
    # Deep neural network training is complicated. The first thing is making
    # sure you have everything ready for training, like datasets, checkpoints,
    # logs, etc. Modify these paths to suit your needs.

    # Datasets
    wider_dir = "/home/robin/data/face/wider"

    # Checkpoint is used to resume training.
    checkpoint_dir = "./checkpoints"

    # Save the model for inference later.
    export_dir = "./exported"

    # Log directory will keep training logs like loss/accuracy curves.
    log_dir = "./logs"

    # All sets. Now it's time to build the model. This model is defined in the
    # `network` module with TensorFlow's functional API.
    model = blaze_net(input_shape=(128, 128, 3))

    # Model built. Restore the latest model if checkpoints are available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")

    # If the restored model is ready for inference, save it and quit training.
    if args.export_only:
        if latest_checkpoint is None:
            print("Warning: Model not restored from any checkpoint.")
        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir)
        print("Model saved at: {}".format(export_dir))
        quit()

    # Construct a dataset for evaluation. Wider face do not provide boxes for
    # test set, use validation data instead.
    dataset_test = build_dataset_from_wider(
        wider_dir, 'wider-test', False, args.batch_size, False, tf.data.experimental.AUTOTUNE)

    # If only evaluation is required.
    if args.eval_only:
        model.evaluate(dataset_test)
        quit()

    # Construct dataset for validation. The loss value from this dataset will be
    # used to decide which checkpoint should be preserved.
    dataset_val = build_dataset_from_wider(
        wider_dir, 'wider-val', False, args.batch_size, False, tf.data.experimental.AUTOTUNE).take(300)

    # Compile the model and print the model summary.
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=BlazeLoss())
    model.summary()

    # Finally, it's time to train the model.

    # Set hyper parameters for training.
    epochs = args.epochs
    batch_size = args.batch_size

    # All done. The following code will setup and start the trainign.

    # Save a checkpoint. This could be used to resume training.
    checkpoint_path = os.path.join(checkpoint_dir, "blazeface")
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_best_only=True)

    # Visualization in TensorBoard
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1024,
                                                       write_graph=True,
                                                       update_freq='epoch')
    callback_image = LogImages(log_dir, "/home/robin/Pictures/family.jpg")

    # List all the callbacks.
    callbacks = [callback_checkpoint, callback_tensorboard, callback_image]

    # Construct training datasets.
    dataset_train = build_dataset_from_wider(
        wider_dir, 'wider-train', True, batch_size, True, tf.data.experimental.AUTOTUNE)

    # Start training loop.
    model.fit(dataset_train, validation_data=dataset_val,
              epochs=epochs, callbacks=callbacks,
              initial_epoch=args.initial_epoch)

    # Make a full evaluation after training.
    model.evaluate(dataset_test)
