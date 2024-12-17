import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
import streamlit as st

#___________________________________________________________________________________________

def plot_latent_space(z_mean, y_test):

    num_dimensions = z_mean.shape[1]

    if num_dimensions == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Digit Label')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization')
        st.pyplot(plt)

    elif num_dimensions == 3:
        fig = px.scatter_3d(x=z_mean[:, 0], y=z_mean[:, 1], z=z_mean[:, 2], color=y_test,
                            labels={'color': 'Digit Label'}, opacity=0.5)
        fig.update_layout(title='Latent Space Visualization',
                          scene=dict(xaxis_title='Latent Dimension 1',
                                     yaxis_title='Latent Dimension 2',
                                     zaxis_title='Latent Dimension 3'))
        st.plotly_chart(fig)

    else:
        pca = PCA(n_components=3)
        z_reduced = pca.fit_transform(z_mean)
        
        fig = px.scatter_3d(x=z_reduced[:, 0], y=z_reduced[:, 1], z=z_reduced[:, 2], color=y_test,
                            labels={'color': 'Digit Label'}, opacity=0.5)
        fig.update_layout(title='Latent Space Visualization- dim >3 so PCA is used',
                          scene=dict(xaxis_title='PCA Dimension 1',
                                     yaxis_title='PCA Dimension 2',
                                     zaxis_title='PCA Dimension 3'))
        st.plotly_chart(fig)