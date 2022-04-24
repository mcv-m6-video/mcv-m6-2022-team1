import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap


def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.close()


def plot_embedding_space(
        embeddings: np.ndarray,
        labels: np.ndarray,
        epoch: int,
        out_path: str = None
):
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(embeddings, y=labels)

    # Plot
    plt.figure(dpi=300, figsize=(15, 15))
    plt.title(f"UMAP of the embedding space at epoch {epoch}")
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="plasma",
        label="Embeddings"
    )

    # for ii, label in enumerate(labels):
    #     plt.annotate(
    #         label,
    #         (embeddings[ii, 0], embeddings[ii, 1]),
    #         alpha=0.5
    #     )
    # plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()
