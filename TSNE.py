from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    # RGB
    color0 = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0)
    color1 = (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0)
    color2 = (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0)
    color3 = (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0)
    color4 = (1.0, 0.4980392156862745, 0.0, 1.0)
    color5 = (1.0, 1.0, 0.2, 1.0)
    color6 = (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0)
    color7 = (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0)
    color8 = (0.6, 0.6, 0.6, 1.0)
    color9 = (0.3, 0.5, 0.4, 0.7)
    colorcenters = (0.1, 0.1, 0.1, 1.0)
    #c = [color0, color1, color2, color3, color4, color5, color6, color7, color8, color9]  # fixed color of 10 classes
    c = [color0, color1, color2, color3, color4, color5]                                                                                # un-fixed color
    print(np.unique(label))
    for i in range(len(np.unique(label))):
        c.append((np.random.random(), np.random.random(), np.random.random(), 1.0))
    # print(len(c))
    for i in range(data.shape[0]):
        color = c[int(label[i])]
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=color,  # plt.cm.Set123
                 fontdict={'weight': 'bold', 'size': 9})
    # plt.legend()
    plt.xlim(-0.005, 1.02)
    plt.ylim(-0.005, 1.02)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def TSNE_PLOT(Z, Y, name="xxx"):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    F = tsne.fit_transform(Z)
    fig1 = plot_embedding(F, Y, name)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()
    #fig1.savefig("images/" + name + ".png", format='png', transparent=True, dpi=1000, pad_inches=0)
