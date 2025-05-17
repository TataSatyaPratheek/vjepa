import umap
import matplotlib.pyplot as plt

def visualize_latent(features, labels):
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(features)
    
    plt.scatter(embeddings[:,0], embeddings[:,1], c=labels)
    plt.colorbar()
    plt.savefig("latent_space.png")
