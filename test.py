import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from patchify import patchify
import tensorflow as tf
from train import load_data, tf_dataset
from vit import VisionTransformer


hp = {}

hp['image_size'] = 200
hp['num_channels'] = 3
hp['patch_size'] = 25
hp['num_patches'] = (hp['image_size'] // hp['patch_size']) ** 2
hp['flatten_patches'] = (hp['num_patches'], hp['patch_size'] * hp['patch_size'] * hp['num_channels'])

hp['batch_size'] = 32
hp['num_epochs'] = 10
hp['lr'] = 1e-4
hp['num_classes'] = 5
hp['class_names'] = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

hp['num_layers'] = 12
hp['hidden_dim'] = 768
hp['mlp_dim'] = 3072
hp['num_heads'] = 12
hp['dropout_rate'] = 0.1

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    dataset_path = "C:/Users/faiss/Downloads/flower_photos/flower_photos"
    model_path = os.path.join('files', 'model.keras')

    train_x , valid_x, test_x = load_data(dataset_path)
    print(f"Number of training images: {len(train_x)}", f"Number of validation images: {len(valid_x)}", f"Number of testing images: {len(test_x)}")
    test_x_ds = tf_dataset(test_x, batch_size=hp['batch_size'])

    model = VisionTransformer(hp)
    model.load_weights(model_path)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp['lr'], clipvalue=1.0),
        metrics=["accuracy"]
    )
    model.evaluate(test_x_ds)