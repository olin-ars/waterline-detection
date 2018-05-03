#!/usr/bin/env python

#   Python libraries
import random

#   Outside libraries
from PIL import Image
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as ptch
import sys

#   Recursion is fun!
sys.setrecursionlimit(10000)

def tuple_diff(tup_1, tup_2):
    """ Returns the difference between each element of two
    tuples of the same length, as a tuple. Tuple items must
    be integers or floats. """

    #   Make sure tuples are same length
    l1 = len(tup_1)
    l2 = len(tup_2)
    assert l1 == l2

    #   Return differnce of each element
    return tuple([abs(tup_1[i] - tup_2[i]) for i in range(l1)])

def img_der(im, vertical = True, tuples = True):
    """ Returns the derivative of a PIL image object, iterating
    through pixels vertically OR horizontally. """

    output = []

    #   Rotate image to get vertical derivative
    if vertical:
        output_image = im.rotate(90, expand=1)


    #   Iterate vertical pixels, getting differences between adjacent pixels
    for y in range(output_image.height):

        #   Reset topmost pixels so you don't get get border pixels
        last = 0

        #   Iterate among horizontal pixels
        for x in range(output_image.width):
            cur = output_image.getpixel((x, y))

            #   Do tuple subtraction if arguments are tuples
            diff = abs(last - cur)
            if x < 2 or x > output_image.width - 2:
                #   Erase boundary points if they are too close to edge
                diff = 0
            last = cur
            output.append(diff)

    #   Apply output data to each pixel of output_image, and rotate it back
    output_image.putdata(tuple(output))
    if vertical:
        output_image = output_image.rotate(-90, expand=1)

    return output_image

def array_to_points(arr, dim):
    """ Takes a length w*h list, where w and h are the width and height of the
    original image, and outputs points.

    Returns a tuple (x, y), with the x and y values of all instances of True
    in the input array."""

    width = dim[0]
    height = dim[1]

    points = []

    #   Iterate through array and find values greater than 1
    for y in range(height):
        for x in range(width):
            cur_val = arr[width*y + x]
            if cur_val:
                if cur_val != (0, 0, 0):
                    points.append((x, y))

    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    #   Y values are measured downward from top left corner

    # plt.show()

    return xs, ys

def ransac(im, trials=200, threshold = 10, plot=False):
    """ Runs ransac on an image, sampling 'points' number of points 'trials'
    number of times. Returns a slope and intercept of line of best fit. """

    #   Convert to boolean image --- pixels are either 1 (white) or 0 (black)
    img = img_der(im, tuples = False).convert(mode="1")
    data = img.getdata()
    xs, ys = array_to_points(list(data), img.size)

    if plot:
        plt.plot(xs, [y for y in ys], 'r.')

    #   Number of points in array
    pnum = len(xs)

    #   Maximum number of points found with ransac
    max_good_points = 0
    good_line = (0, 0)

    #   Perform ransac 'trials' number of times and find best line
    for i in range(trials):

        points_to_connect = []

        #   Lines have two points --- theoretically could smooth with more
        for j in range(2):
            idx = int(random.random()*pnum)
            points_to_connect.append((xs[idx], ys[idx]))

        dists = []
        m, b = line_from_points(points_to_connect[0], points_to_connect[1])

        #   Determines how many points fall within a threshold distance of
        #   ransac line
        for k in range(pnum):
            x = xs[k]
            y = ys[k]
            d = dist_point_to_line((x, y), (m, b))
            if d <= threshold:
                dists.append(d)

        #       Keep track of best line so far
        if len(dists) > max_good_points:
            max_good_points = len(dists)
            good_line = (m, b)

    print(good_line)
    return good_line


def dist_point_to_line(p, l):
    #   TODO make a more effective distance function than vertical distance
    return abs(p[1] - (l[0]*p[0] + l[1]))

def line_from_points(p1, p2):
    """ Returns tuple (m, b) from two points, where m is the slope and b is the
    y-intercept of the line that connects them. """
    #   TODO Make this find line of best fit of several points?

    try:
        m = float(p2[1] - p1[1])/(p2[0] - p1[0])
        b = p1[1] - m*p1[0]
    except ZeroDivisionError:
        m, b = (0, 0)

    return (m, b)

def get_obstacle_pixels(im, plot=False):
    """ Takes an image and gets a list of pixels that are below the waterline
    not water. """

    im = im.convert(mode="L", dither=None)
    data = im.getdata()
    md = max(data)
    data = [117 - item for item in data]


    #   Find values of x and y for pixels above the actual waterline
    baw_xs, baw_ys = array_to_points(list(data), im.size)
    baw_ys = [im.height-y for y in baw_ys]

    m, b = ransac(im)
    abv_threshold = 2   #   Margin, in pixels, around waterline where obstacles
                        #   don't count.

    obs_xs = []
    obs_ys = []
    for i, x in enumerate(baw_xs):
        y = baw_ys[i]
        if -m*x + (im.height - b) - abv_threshold >= y:
            obs_xs.append(x)
            obs_ys.append(y)


    if plot: plt.plot(obs_xs, obs_ys, '.', markersize=1, alpha=0.4)
    return (obs_xs, obs_ys)


def find_clusters(xs, ys, v=True):
    """ Separates points into clusters that are connected.

    Takes a list of x values and a list of y values, and returns a set of four-
    integer tuples (top, bottom, left, right), where top is the highest y
    value, right is the highest x value, and so on.

    Recursively find objects in map, checking the eight directionally adjacent
    pixels.
    """

    objs = []
    usedset = set()
    pixel_threshold = 1
    #objects.append(Set)

    img = np.zeros((max(xs) + 1, max(ys) + 1))
    for i, x in enumerate(xs):
        img[x, ys[i]] = 1


    #   Choose the first point to seed the cluster
    for i, ref_x in enumerate(xs):
        #print(ref_x)

        ref_y = ys[i];
        if ((ref_x, ref_y) in usedset):
            continue
        else:
            tmpset = set() #Start a new object set

            #Use find_clusters_recurs to populate the set with pixels in the object
            find_clusters_recurs(ref_x, ref_y, img, tmpset, usedset)

            if len(tmpset) > pixel_threshold: #TODO: FINISH ERROR CHECKING
                if v: print(surprise() + " Found object with %i pixels in it." % len(tmpset))
                objs.append(tmpset) #Put this new object set into the list of objects

    if v: print("%i total objects found" % len(objs))

    boundboxes = set()

    #   Iterate through identified objects
    for s in objs:

        xs = [t[0] for t in s]
        ys = [t[1] for t in s]

        #   Choose default values guarenteed to not be most extreme coordinates
        top = max(ys)
        bottom = min(ys)
        left = min(xs)
        right = max(xs)
        width = right - left
        height = top - bottom

        #   Add rectangle to valid objects
        boundboxes.add((left, bottom, width, height)) #Tuple for bounding box

    return boundboxes


def find_clusters_recurs(x, y, image, currset, usedset):
    """ Recursively updates set objects with adjacent x and y values. """

    usedset.add((x, y)) #Add this node to the used set
    currset.add((x, y)) #Add this node to current object set

    #   List of pixel values to check, relative to current pixel
    try_x = [0,1,1,1,0,-1,-1,-1]
    try_y = [1,1,0,-1,-1,-1,0,1]

    for i in range(len(try_x)):

        test_x = x + try_x[i]
        test_y = y + try_y[i]

        #   Makes sure test pixels are within frame of image
        if test_x > image.shape[0] - 1 or test_x < 0:
            continue
        if test_y > image.shape[1] - 1 or test_y < 0:
            continue

        #   Adds adjacent pixels if true and not in used set already:
        if (test_x, test_y) not in usedset:
            if (image[test_x][test_y] == 1):
                #   Who doesn't love a little bit of recursion?
                #   Python, turns out.
                find_clusters_recurs(test_x, test_y, image, currset, usedset)


def draw_rectangle(ax, xywh, alpha=0.5, color="yellow"):
    """ Draws a rectangle with specified width, height, and lower left corner.
    Takes in an axis to draw on and a tuple (x, y, w, h). """

    x, y, width, height = xywh
    ax.add_patch(ptch.Rectangle((x, y), width, height, alpha=alpha, facecolor=color, linewidth=0))


def surprise():
    """ This is a helper function of utmost importance.
    Returns a random exclamation of surprise as a string. """

    phrases = ["Wowsers!", "Woohoo!", "What are the chances!", "No way!",
        "Hell yeah!", "Lit!", "Success!", "Hahaha!"]
    return random.choice(phrases)


def ransac_plot(original_image, waterline_image):
    """ Visualization for ransac waterline detection. Inputs are image paths to
    the original image and another annotated image with pixels for water. """

    #   Open image files
    try:
        pyp_image = mpimg.imread(original_image)
        im = Image.open(waterline_image)
    except IOError:
        print("RANSAC plot: One or more image files could not be found.")
        return

    obs_xs, obs_ys = get_obstacle_pixels(im, plot=True)
    clusters = find_clusters(obs_xs, obs_ys, v=0)

    #   Flip image so it displays right-side up.
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im = im.convert(mode="L", dither=None)

    #   Run ransac
    m, b = ransac(im, plot=False)

    #   Show the image, edge detection, and line of best fit
    ax = plt.gca()
    plt.imshow(np.flipud(pyp_image), origin='lower')
    plt.plot([0, 1280], [m*x + b for x in [0, 1280]], 'w', linewidth=2.0)
    for rectangle in clusters:
        print(rectangle)
        draw_rectangle(ax, (rectangle[0], 0, rectangle[2], 722), color="white", alpha=0.2)
        draw_rectangle(ax, rectangle)
    plt.xlim([0, 1280])
    plt.ylim([0, 722])
    plt.show()


if __name__ == '__main__':
    img_num = 1711
    file = "Test/%s.0.png" % img_num
    file_train = "Test/%s.0-annotation.png" % img_num
    ransac_plot(file, file_train)
