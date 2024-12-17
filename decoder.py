from tensorflow.keras.layers import Dense, Reshape, Input  #even if displayed as error, it is a know issue of lazy importation, the code will execute anyway
from tensorflow.keras.models import Model

#___________________________________________________________________________________________

def build_decoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(256, activation='relu')(latent_inputs)
    x = Dense(28 * 28, activation='sigmoid')(x)
    outputs = Reshape((28, 28, 1))(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder