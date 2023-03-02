import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import webcolors


# Define the list of complex color names and their simplified equivalents

COLORS = {
    "aliceblue": "light blue",
    "antiquewhite": "off-white",
    "aqua": "cyan",
    "aquamarine": "light green",
    "azure": "light blue",
    "beige": "tan",
    "bisque": "orange",
    "black": "black",
    "blanchedalmond": "tan",
    "blue": "blue",
    "blueviolet": "purple",
    "brown": "brown",
    "burlywood": "tan",
    "cadetblue": "blue",
    "chartreuse": "green",
    "chocolate": "brown",
    "coral": "orange",
    "cornflowerblue": "blue",
    "cornsilk": "off-white",
    "crimson": "red",
    "cyan": "cyan",
    "darkblue": "blue",
    "darkcyan": "cyan",
    "darkgoldenrod": "brown",
    "darkgray": "gray",
    "darkgrey": "gray",
    "darkgreen": "green",
    "darkkhaki": "tan",
    "darkmagenta": "purple",
    "darkolivegreen": "green",
    "darkorange": "orange",
    "darkorchid": "purple",
    "darkred": "red",
    "darksalmon": "pink",
    "darkseagreen": "green",
    "darkslateblue": "blue",
    "darkslategray": "gray",
    "darkslategrey": "gray",
    "darkturquoise": "cyan",
    "darkviolet": "purple",
    "deeppink": "pink",
    "deepskyblue": "blue",
    "dimgray": "gray",
    "dimgrey": "gray",
    "dodgerblue": "blue",
    "firebrick": "red",
    "floralwhite": "off-white",
    "forestgreen": "green",
    "fuchsia": "purple",
    "gainsboro": "gray",
    "ghostwhite": "off-white",
    "gold": "yellow",
    "goldenrod": "brown",
    "gray": "gray",
    "grey": "gray",
    "green": "green",
    "greenyellow": "green",
    "honeydew": "off-white",
    "hotpink": "pink",
    "indianred": "red",
    "indigo": "purple",
    "ivory": "off-white",
    "khaki": "tan",
    "lavender": "purple",
    "lavenderblush": "pink",
    "lawngreen": "green",
    "lemonchiffon": "off-white",
    "lightblue": "light blue",
    "lightcoral": "pink",
    "lightcyan": "light blue",
    "lightgoldenrodyellow": "yellow",
    "lightgray": "light gray",
    "lightgrey": "light gray",
    "lightgreen": "light green",
    'lightgoldenrodyellow': 'yellow',
    'lightgray': 'gray',
    'lightgreen': 'green',
    'lightgrey': 'gray',
    'lightpink': 'pink',
    'lightsalmon': 'orange',
    'lightseagreen': 'green',
    'lightskyblue': 'blue',
    'lightslategray': 'gray',
    'lightslategrey': 'gray',
    'lightsteelblue': 'blue',
    'lightyellow': 'yellow',
    'lime': 'green',
    'limegreen': 'green',
    'linen': 'beige',
    'magenta': 'pink',
    'maroon': 'brown',
    'mediumaquamarine': 'green',
    'mediumblue': 'blue',
    'mediumorchid': 'purple',
    'mediumpurple': 'purple',
    'mediumseagreen': 'green',
    'mediumslateblue': 'blue',
    'mediumspringgreen': 'green',
    'mediumturquoise': 'blue',
    'mediumvioletred': 'red',
    'midnightblue': 'blue',
    'mintcream': 'white',
    'mistyrose': 'pink',
    'moccasin': 'yellow',
    'navajowhite': 'beige',
    'navy': 'blue',
    'oldlace': 'beige',
    'olive': 'green',
    'olivedrab': 'green',
    'orange': 'orange',
    'orangered': 'red',
    'orchid': 'purple',
    'palegoldenrod': 'yellow',
    'palegreen': 'green',
    'paleturquoise': 'blue',
    'palevioletred': 'red',
    'papayawhip': 'beige',
    'peachpuff': 'orange',
    'peru': 'brown',
    'pink': 'pink',
    'plum': 'purple',
    'powderblue': 'blue',
    'purple': 'purple',
    'red': 'red',
    'rosybrown': 'brown',
    'royalblue': 'blue',
    'saddlebrown': 'brown',
    'salmon': 'orange',
    'sandybrown': 'orange',
    'seagreen': 'green',
    'seashell': 'beige',
    'sienna': 'brown',
    'silver': 'gray',
    'skyblue': 'blue',
    'slateblue': 'blue',
    'slategray': 'gray',
    'slategrey': 'gray',
    'snow': 'white',
    'springgreen': 'green',
    'steelblue': 'blue',
    'tan': 'brown',
    'teal': 'green',
    'thistle': 'purple',
    'tomato': 'red',
    'turquoise': 'blue',
    'violet': 'purple',
    'wheat': 'beige',
    'white': 'white',
    'whitesmoke': 'gray',
    'yellow': 'yellow',
    'yellowgreen': 'green'


}


def closest_color(requested_color):
    """Maps an RGB tuple to the closest human-friendly color name"""
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_top_colors(image, k=4, n=1):
    """Finds the top n dominant colors in the input image using K-means clustering"""
    # Convert the image to a numpy array
    pixels = np.array(image)
    # Reshape the array to a 2D array of pixels
    pixels = pixels.reshape(-1, 3)
    # Apply K-means clustering to find the dominant colors
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(pixels)
    # Get the colors of the cluster centers
    colors = kmeans.cluster_centers_
    # Get the count of pixels assigned to each cluster
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    # Sort the colors by frequency
    sorted_colors = colors[np.argsort(-counts)]
    # Convert the color values to integers and map to color names
    color_names = [closest_color(np.round(color).astype(int))
                   for color in sorted_colors]

    # Return the top n colors
    top_colors = color_names[:n]
    simpleColors = []

    try:
        for color in top_colors:
            simpleColors.append(COLORS[color])
    except:
        print(COLORS[color])

    top_colors += (simpleColors)
    top_colors = list(set(top_colors))
    return top_colors


# Load the image
image = Image.open("./in.jpg")

# Get the top 3 dominant colors
top_colors = get_top_colors(image, k=4, n=5)
# Print the top colors
print(f"The top colors are {', '.join(top_colors)}")
