import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt


class Visualize:

  def __init__(self):
    self.colours = ['r', 'b', 'g']
    self.cl = 0

  def plot(self, xs, ys, labels, colours=None):
    """Plots the points xs,ys with labels drawn on each point"""
    plt.scatter(xs, ys, c=colours)
    if labels is not None:
      for label, x, y in zip(labels, xs, ys):
        plt.annotate(
          label,
          xy=(x, y), xytext=(-30, 30),
          textcoords='offset points', ha='right', va='bottom',
          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    self.cl += 1

  def plot_cluster(self, centroids):
    """Plots the centroids of the cluster"""
    self.plot(centroids[:, 0], centroids[:, 1], labels=None, colours=['g'] * centroids.shape[1])

  def plot_documents(self, svd, names, doc_clusters, no_clusters):
    """Plots documents on the maps (similar documents clustered together"""
    u, vt = svd
    pts = vt
    # each cluster gets a different colour
    colormap = plt.get_cmap("hsv")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=no_clusters)
    scalarMap = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    self.plot(pts[1], pts[2], names, colours=[scalarMap.to_rgba(i) for i in doc_clusters])

  def show(self):
    """Show the plot"""
    plt.axis((-1, 1, 1, -1,))
    plt.show()