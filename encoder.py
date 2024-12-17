from tensorflow.keras.layers import Input, Dense, Lambda, Flatten  #even if displayed as error, it is a know issue of lazy importation, the code will execute anyway
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

#___________________________________________________________________________________________

def build_encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder
