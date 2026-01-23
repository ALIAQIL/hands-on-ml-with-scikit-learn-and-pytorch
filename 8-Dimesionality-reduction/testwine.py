from sklearn.datasets import fetch_openml
import umap.umap_ as umap
import plotly.express as px

mnist = fetch_openml('mnist_784',as_frame=False)
features = mnist["data"]
labels = mnist["target"]
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean')
umap_3d = reducer.fit_transform(features)

fig = px.scatter_3d(
    umap_3d, x=0, y=1, z=2,
    color=labels,
    labels={'color': 'digit'},
    title='MNIST 3D UMAP Projection',
    opacity=0.5,
    size_max=1
)

fig.update_traces(marker=dict(size=2))
fig.show()