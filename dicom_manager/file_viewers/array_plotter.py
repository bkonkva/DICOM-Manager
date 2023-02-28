from matplotlib.patches import Patch
from matplotlib import pyplot as plt


class ArrayPlotter:

    zoom: int = 1
    legend: bool = False
    legend_position: str = "upper left"
    legend_size: int = 1

    #     def __init__(self)

    def add_legend(self, ax, legend):
        legend_elements = [
            Patch(facecolor=legend[j][1], edgecolor=legend[j][1], label=legend[j][0])
            for j in range(len(legend))
        ]
        return ax.legend(
            handles=legend_elements,
            loc=self.legend_position,
            prop={"size": self.legend_size * self.zoom},
        )

    def read_kwargs(self, **kwargs) -> None:
        if "zoom" in kwargs:
            self.zoom = kwargs["zoom"]
        if "legend" in kwargs:
            self.legend = kwargs["legend"]
        if "legend_size" in kwargs:
            self.legend_size = kwargs["legend_size"]
        if "legend_position" in kwargs:
            self.legend_position = kwargs["legend_position"]

    def plot_images(self, image_list, **kwargs) -> None:
        self.read_kwargs(**kwargs)
        fig = plt.figure(figsize=(15 * self.zoom, 15 * self.zoom), dpi=64)
        for i in range(1, (1 + len(image_list))):
            ax = fig.add_subplot(1, len(image_list), i)
            if self.legend:
                ax = self.add_legend(ax, self.legend)
            plt.imshow(image_list[i - 1], cmap="gray")
