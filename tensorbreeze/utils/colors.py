import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    label = int(label)

    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


"""
Generated using:
```
colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 90)]
shuffle(colors)
pprint(colors)
```
"""
colors = [
    [255, 0, 119],
    [16, 255, 0],
    [255, 0, 220],
    [0, 255, 203],
    [255, 0, 238],
    [135, 255, 0],
    [0, 255, 85],
    [0, 255, 51],
    [0, 255, 221],
    [169, 255, 0],
    [0, 51, 255],
    [255, 50, 0],
    [255, 85, 0],
    [153, 255, 0],
    [135, 0, 255],
    [255, 0, 84],
    [0, 153, 255],
    [255, 221, 0],
    [0, 255, 135],
    [221, 0, 255],
    [0, 255, 33],
    [170, 0, 255],
    [51, 255, 0],
    [255, 136, 0],
    [204, 0, 255],
    [187, 0, 255],
    [0, 255, 67],
    [0, 255, 238],
    [0, 255, 0],
    [255, 238, 0],
    [102, 0, 255],
    [0, 186, 255],
    [0, 203, 255],
    [255, 0, 152],
    [255, 204, 0],
    [203, 255, 0],
    [255, 0, 169],
    [255, 67, 0],
    [101, 255, 0],
    [16, 0, 255],
    [221, 255, 0],
    [0, 255, 102],
    [255, 170, 0],
    [187, 255, 0],
    [255, 119, 0],
    [255, 0, 67],
    [255, 0, 203],
    [238, 255, 0],
    [0, 16, 255],
    [255, 0, 16],
    [238, 0, 255],
    [255, 0, 102],
    [255, 255, 0],
    [0, 255, 119],
    [0, 67, 255],
    [255, 0, 33],
    [0, 255, 16],
    [153, 0, 255],
    [0, 119, 255],
    [119, 0, 255],
    [0, 33, 255],
    [0, 255, 255],
    [255, 0, 135],
    [67, 0, 255],
    [0, 255, 170],
    [255, 0, 0],
    [255, 0, 187],
    [0, 220, 255],
    [255, 16, 0],
    [33, 0, 255],
    [33, 255, 0],
    [255, 33, 0],
    [0, 255, 187],
    [0, 255, 153],
    [255, 153, 0],
    [67, 255, 0],
    [51, 0, 255],
    [84, 0, 255],
    [84, 255, 0],
    [0, 237, 255],
    [0, 0, 255],
    [255, 102, 0],
    [255, 187, 0],
    [0, 169, 255],
    [0, 84, 255],
    [0, 102, 255],
    [255, 0, 51],
    [255, 0, 255],
    [0, 135, 255],
    [118, 255, 0]
]
