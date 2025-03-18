import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
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

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split_ratio=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.jpg")))
    split_size = int(len(images) * split_ratio)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    return train_x, valid_x, test_x

def process_image(image_path):
    image_path = image_path.decode()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (hp['image_size'], hp['image_size']))
    image = image / 255.0

    patch_shape = (hp['patch_size'], hp['patch_size'], hp['num_channels'])
    patches = patchify(image, patch_shape, hp['patch_size'])
    patches = np.reshape(patches, hp['flatten_patches'])
    patches = patches.astype(np.float32)
    class_name = os.path.basename(os.path.dirname(image_path))
    class_index = hp['class_names'].index(class_name)
    class_index = np.array(class_index, dtype=np.int32)
    return patches, class_index

def parse(image_path):
    patches, labels = tf.numpy_function(process_image, [image_path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp['num_classes'])
    patches.set_shape(list(hp['flatten_patches']))  # Convertir le tuple en liste
    labels.set_shape([hp['num_classes']])
    return patches, labels

def tf_dataset(images, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch_size).prefetch(8)
    return ds

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir('files')
    dataset_path = "C:/Users/faiss/Downloads/flower_photos/flower_photos"
    model_path = os.path.join('files', 'model.keras')
    csv_path = os.path.join('files', 'log.csv')

    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Number of training images: {len(train_x)}", f"Number of validation images: {len(valid_x)}", f"Number of testing images: {len(test_x)}")
    train_ds = tf_dataset(train_x, batch_size=hp['batch_size'])
    valid_ds = tf_dataset(valid_x, batch_size=hp['batch_size'])

    model = VisionTransformer(hp)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp['lr'], clipvalue=1.0),
        metrics=["accuracy"]
    )
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]
    model.fit(train_ds, epochs=hp['num_epochs'], validation_data=valid_ds, callbacks=callbacks)