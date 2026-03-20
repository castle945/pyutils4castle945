def sklearn_digits():
    from sklearn import datasets
    digits = datasets.load_digits(n_class=6) # MNIST test set 的子集，10 类总样本数 1797, 特征维度 8x8=64
    data, label, (n_samples, n_features) = digits.data, digits.target, digits.data.shape
    datadb.set('sklearn/digits', [data, label])

# @Todo 待删除，加到 codenotes 仓库里就行
@rpc_func
def plot_tsne2d(
    features: np.ndarray, labels: np.ndarray,
    x: str = 'x', y: str = 'y', title: str = 'T-SNE',
    rpc: bool = False,
) -> None:
    """
    Args:
        features (ndarray(N, M)): N 个归一化的样本，每个 M 维
        labels (ndarray(N,)): 聚类标签
    """
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd
    import seaborn
    import matplotlib.pyplot as plt

    if features.shape[1] > 2:
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(features)

    df = pd.DataFrame({'x': features[:, 0], 'y': features[:, 1], 'label': labels})
    seaborn.scatterplot(
        data=df, x=x, y=y, hue=df.label, 
        palette=seaborn.color_palette("hls", len(np.unique(labels))),
    ).set(title=title)
    plt.show()
@rpc_func
def plot_umap(
    features: np.ndarray, labels: np.ndarray,
    x: str = 'x', y: str = 'y', title: str = 'UMap',
    rpc: bool = False,
) -> None:
    # 与 t-SNE 相比，它在保持数据全局结构方面更加出色，但更慢
    # see https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_mnist_example.html
    import umap # pip install umap-learn
    import numpy as np
    import pandas as pd
    import seaborn
    import matplotlib.pyplot as plt

    if features.shape[1] > 2:
        features = umap.UMAP(random_state=0).fit_transform(features)

    df = pd.DataFrame({'x': features[:, 0], 'y': features[:, 1], 'label': labels})
    seaborn.scatterplot(
        data=df, x=x, y=y, hue=df.label, 
        palette=seaborn.color_palette("hls", len(np.unique(labels))),
    ).set(title=title)
    plt.show()

@pytest.mark.skip(reason='visualization func')
def test_tsne():
    try:
        import sklearn, umap, seaborn, pandas
    except:
        return
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    data, label = datadb.get('list2/sklearn/digits', None)
    pu4c.cv.plot_tsne2d(data, label, rpc=True)
    # pu4c.cv.plot_umap(data, label, rpc=True) # umap 版本变更暂未适配新版本