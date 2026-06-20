import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_pure_cmap(color_name):
    """Create a pure color colormap with the given base color"""
    base_color = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "pure_red": [1, 0, 0],
        "pure_green": [0, 1, 0],
        "pure_blue": [0, 0, 1],
        "pure_magenta": [1, 0, 1],
        "pure_cyan": [0, 1, 1],
        "pure_yellow": [1, 1, 0],
    }.get(color_name, [1, 0, 0])

    # Create colormap from black to the pure color
    cdict = {
        "red": [(0.0, 0.0, 0.0), (1.0, base_color[0], base_color[0])],
        "green": [(0.0, 0.0, 0.0), (1.0, base_color[1], base_color[1])],
        "blue": [(0.0, 0.0, 0.0), (1.0, base_color[2], base_color[2])],
    }

    return mcolors.LinearSegmentedColormap(color_name, cdict)


class MicroImage:
    """Class to hold information about a microimage"""

    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig


def microshow(
    images,
    cmaps=None,
    fig=None,
    fig_scaling=10,
    flip_map=None,
    label_text=None,
    label_color="white",
):
    """
    Display multiple images with overlay using specified colormaps.

    Parameters:
    -----------
    images : list of 2D numpy arrays
        Images to display
    cmaps : list of str or None
        Colormaps to use for each image. If None, default colormaps will be used.
    fig : matplotlib.figure.Figure or None
        Figure to use. If None, a new figure will be created.
    fig_scaling : float
        Scaling factor for the figure size.
    flip_map : list of bool or None
        Whether to flip the colormap for each image.
    label_text : str or None
        Text to display in the top-left corner.
    label_color : str
        Color of the label text.

    Returns:
    --------
    microim : MicroImage
        Object containing the figure and axes.
    """
    if not isinstance(images, list):
        images = [images]

    # Set default colormaps if not provided
    if cmaps is None:
        cmaps = ["pure_red", "pure_green", "pure_blue"][: len(images)]

    # Set default flip_map if not provided
    if flip_map is None:
        flip_map = [False] * len(images)

    # Create a figure if not provided
    if fig is None:
        # Calculate a reasonable figure size based on image shape and scaling
        sample_shape = images[0].shape
        aspect_ratio = sample_shape[1] / sample_shape[0]
        fig_size = (fig_scaling * aspect_ratio, fig_scaling)
        fig = plt.figure(figsize=fig_size)

    # Create a single subplot
    ax = fig.add_subplot(111)

    # Initialize a blank RGB image
    max_shape = max([img.shape for img in images], key=lambda x: x[0] * x[1])
    composite = np.zeros((*max_shape[:2], 3), dtype=np.float32)

    # Add each image with its colormap
    for i, (img, cmap_name, flip) in enumerate(zip(images, cmaps, flip_map)):
        # Handle different data types and normalization
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            # Normalize to 0-1
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)

        # Ensure 2D grayscale
        if len(img.shape) > 2:
            img = np.mean(img, axis=2)

        # Flip values if specified
        if flip:
            img = 1.0 - img

        # Get colormap
        if cmap_name in [
            "pure_red",
            "pure_green",
            "pure_blue",
            "pure_magenta",
            "pure_cyan",
            "pure_yellow",
        ]:
            cmap = create_pure_cmap(cmap_name)
        else:
            try:
                cmap = plt.get_cmap(cmap_name)
            except Exception:
                cmap = create_pure_cmap("pure_red")

        # Apply colormap to get RGBA
        colored = cmap(img)

        # Add to composite (RGB only, ignore alpha)
        composite += colored[:, :, :3]

    # Clip to valid range
    composite = np.clip(composite, 0, 1)

    # Display the composite image
    ax.imshow(composite)
    ax.axis("off")

    # Add label text if provided
    if label_text is not None:
        ax.text(
            0.05,
            0.95,
            label_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=label_color,
            verticalalignment="top",
        )

    fig.tight_layout()

    # Return a MicroImage object
    return MicroImage(ax, fig)


class Micropanel:
    """Class for creating panels of MicroImages"""

    def __init__(self, rows=1, cols=1, figsize=None):
        """
        Initialize a micropanel with the given number of rows and columns.

        Parameters:
        -----------
        rows : int
            Number of rows in the panel.
        cols : int
            Number of columns in the panel.
        figsize : list of float or None
            Figure size in inches [width, height]. If None, a size will be calculated.
        """
        if figsize is None:
            figsize = [cols * 4, rows * 4]

        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize)
        # Make axes 2D even if rows or cols is 1
        if rows == 1 and cols == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1:
            self.axes = np.array([self.axes])
        elif cols == 1:
            self.axes = np.array([[ax] for ax in self.axes])

        # Turn off all axes initially
        for row in self.axes:
            for ax in row:
                ax.axis("off")

        self.rows = rows
        self.cols = cols
        plt.tight_layout()

    def add_element(self, pos, microim):
        """
        Add a MicroImage to the panel at the specified position.

        Parameters:
        -----------
        pos : list of int [row, col]
            Position in the panel (0-indexed).
        microim : MicroImage
            MicroImage object to add.
        """
        row, col = pos
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise ValueError(
                f"Position {pos} out of bounds for panel of size {self.rows}x{self.cols}"
            )

        ax = self.axes[row][col]

        # Clear the axis
        ax.clear()

        # Copy the image from the MicroImage to this axis
        # Get the image data from the MicroImage's axis
        for image in microim.ax.get_images():
            # Copy the image to the new axis
            ax.imshow(image.get_array(), cmap=image.get_cmap(), norm=image.norm)

        # Copy any text elements
        for text in microim.ax.texts:
            ax.text(
                text.get_position()[0],
                text.get_position()[1],
                text.get_text(),
                fontsize=text.get_fontsize(),
                transform=(
                    ax.transAxes
                    if text.get_transform() == microim.ax.transAxes
                    else None
                ),
                color=text.get_color(),
                ha=text.get_ha(),
                va=text.get_va(),
                fontweight=text.get_weight(),
            )

        # Turn off axis
        ax.axis("off")

    def savefig(self, filename, **kwargs):
        """
        Save the panel to a file.

        Parameters:
        -----------
        filename : str
            Filename to save to.
        **kwargs : dict
            Additional arguments to pass to plt.savefig().
        """
        self.fig.savefig(filename, **kwargs)
