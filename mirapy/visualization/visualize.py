import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # do not remove this line


def visualize_2d(x, y):
    """
    Function for 2D visualization of data using Principal Component Analysis
    (PCA).

    :param x: Array of features.
    :param y: Array of target values.
    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['PC 1', 'PC 2'])
    target = pd.DataFrame(y, columns=['target'])
    new_df = pd.concat([principal_df, target], axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2D Feature Visualization')

    im = ax.scatter(new_df[['PC 1']], new_df[['PC 2']],
                    c=new_df[['target']].values, s=2)
    fig.colorbar(im, ax=ax)

    ax.grid()


def visualize_3d(x, y):
    """
    Function for 3D visualization of data using Principal Component Analysis
    (PCA).

    :param x: Array of features.
    :param y: Array of target values.
    """
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['PC 1', 'PC 2', 'PC 3'])
    target = pd.DataFrame(y, columns=['target'])
    new_df = pd.concat([principal_df, target], axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    im = ax.scatter(new_df[['PC 1']], new_df[['PC 2']], new_df[['PC 3']], c=y,
                    s=2)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D Feature Visualization')
    fig.colorbar(im, ax=ax)
    ax.grid()
