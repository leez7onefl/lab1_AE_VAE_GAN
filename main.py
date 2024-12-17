import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from encoder import build_encoder
from decoder import build_decoder
from plotter import plot_latent_space

#___________________________________________________________________________________________

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    return x_train, y_train, x_test, y_test

#___________________________________________________________________________________________

class VAELossLayer(Layer):
    def __init__(self, loss_function, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.loss_function = loss_function

    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        reconstruction_loss = self.loss_function(K.flatten(x), K.flatten(x_decoded))
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.add_loss(K.mean(reconstruction_loss + kl_loss))
        return x_decoded

#___________________________________________________________________________________________

def custom_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

#___________________________________________________________________________________________

def compile_vae(encoder, decoder, input_shape, optimizer, lr, loss_choice):
    inputs = encoder.input
    z_mean, z_log_var, z = encoder(inputs)
    outputs = decoder(z)
    
    if loss_choice == 'Binary Crossentropy':
        loss_function = binary_crossentropy
    elif loss_choice == 'Mean Squared Error':
        loss_function = custom_mean_squared_error
    
    vae_outputs = VAELossLayer(loss_function)([inputs, outputs, z_mean, z_log_var])
    vae = Model(inputs, vae_outputs, name='vae_mlp')

    optimizers = {
        'Adam': Adam(learning_rate=lr),
        'RMSProp': RMSprop(learning_rate=lr),
        'SGD': SGD(learning_rate=lr)
    }
    selected_optimizer = optimizers[optimizer]

    vae.compile(optimizer=selected_optimizer)
    return vae

#___________________________________________________________________________________________

def plot_images(original, decoded):
    n = 10
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(decoded[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    return fig

#___________________________________________________________________________________________

def save_models(encoder, decoder):
    """Save encoder and decoder models' weights."""
    encoder.save_weights('vae_encoder.weights.h5') 
    decoder.save_weights('vae_decoder.weights.h5')

#___________________________________________________________________________________________

def load_decoder(latent_dim):
    """Load the decoder model with its trained weights."""
    decoder = build_decoder(latent_dim)
    decoder.load_weights('vae_decoder.weights.h5')  
    return decoder

#___________________________________________________________________________________________

def vae_training(latent_dim):
    st.title("Lab1 - VAE & GAN")

    st.sidebar.title("Hyperparameters")
    epochs = st.sidebar.slider("Epochs", 1, 100, 50)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=0)
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "RMSProp", "SGD"])
    learning_rate = float(st.sidebar.text_input("Learning Rate", "1e-3"))
    loss_choice = st.sidebar.selectbox("Loss Function", ["Binary Crossentropy", "Mean Squared Error"])

    if st.sidebar.button("Start Training"):
        input_shape = (28, 28, 1)
        x_train, y_train, x_test, y_test = load_data()
        encoder = build_encoder(input_shape, latent_dim)
        decoder = build_decoder(latent_dim)
        vae = compile_vae(encoder, decoder, input_shape, optimizer, learning_rate, loss_choice)

        history = {'loss': []}

        for epoch in range(epochs):
            hist = vae.fit(x_train, epochs=1, batch_size=batch_size, validation_data=(x_test, None), verbose=0)
            history['loss'].append(hist.history['loss'][0])

            x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
            x_test_decoded = decoder.predict(x_test_encoded[2], batch_size=batch_size)
            
            st.subheader(f"Epoch {epoch + 1}/{epochs}")
            st.write(f"Loss: {hist.history['loss'][0]:.4f}")
            
            fig = plot_images(x_test, x_test_decoded)
            st.pyplot(fig)
            
            plot_latent_space(x_test_encoded[0], y_test)

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), history['loss'], label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            st.pyplot(plt)
        
        save_models(encoder, decoder)

#___________________________________________________________________________________________

def generate_random_image(latent_dim):    
    with st.spinner('Loading model...'):
        decoder = load_decoder(latent_dim)
    st.success('Model loaded.')

    z_sample = np.random.normal(size=(1, latent_dim))
    generated_image = decoder.predict(z_sample)
    st.image(generated_image.reshape(28, 28), width=512, caption="Generated Image")

#___________________________________________________________________________________________

def main():
    page = st.sidebar.selectbox("", ["VAE Training", "GAN Generation (after VAE training)"])
    latent_dim = st.sidebar.slider("Latent Dimension", 2, 10, 2)

    if page == "VAE Training":
        vae_training(latent_dim)
    elif page == "GAN Generation (after VAE training)":
        st.title("VAE Image Generation")
        if st.sidebar.button("Start Image Generation"):
            generate_random_image(latent_dim)

#___________________________________________________________________________________________

if __name__ == "__main__":
    main()