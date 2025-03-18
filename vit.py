import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, config):
    x = Dense(config['mlp_dim'], activation=tf.nn.relu)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(config['hidden_dim'])(x)
    x = Dropout(config['dropout_rate'])(x)
    return x


def transformer_encoder_block(x, config):
    skip_l = x
    x = LayerNormalization()(x)
    x  = MultiHeadAttention(
        num_heads=config['num_heads'],
        key_dim=config['hidden_dim']
    )(x,x)
    x = Add()([x, skip_l])
    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, config)
    x = Add()([x, skip_2])  
    return x



def VisionTransformer(config):
    inputs_shape = (config['num_patches'], config['patch_size'] *config['patch_size'] *config['num_channels'])
    inputs = Input(inputs_shape) #  (256, 3072)
    patch_embedding = Dense(config['hidden_dim'])(inputs)  # (256, 768)
    position= tf.range(start=0 , limit=config['num_patches'], delta=1)
    position_embedding = Embedding(input_dim=config['num_patches'], output_dim=config['hidden_dim'])(position)
     # (256, 768)
    embed = patch_embedding + position_embedding

    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed]) # (257, 768)

    for _ in range(config['num_layers']):
        x = transformer_encoder_block(x, config)

    x = LayerNormalization()(x)
    x = x[: , 0 , :]
    x = Dense(config['num_classes'], activation=tf.nn.softmax)(x)
    model = Model(inputs=inputs, outputs=x)
    return model


if __name__ == "__main__":
    config = {}
    config['num_layers'] = 12
    config['hidden_dim'] = 768
    config['mlp_dim'] = 3072
    config['num_heads'] = 12
    config['dropout_rate'] = 0.1
    config['num_patches'] = 256
    config['patch_size'] = 32
    config['num_channels'] = 3
    config['num_classes'] = 5
    
    model = VisionTransformer(config)
    model.summary()