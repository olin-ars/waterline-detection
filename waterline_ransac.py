#!/usr/bin/env python

#   Python libraries
import random

#   Outside libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
                points.append((x, y))

    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    #   Y values are measured downward from top left corner

    plt.plot(xs, [y for y in ys], 'r.')
    # plt.show()

    return xs, ys

def ransac(im, trials=200, threshold = 30):
    """ Runs ransac on an image, sampling 'points' number of points 'trials'
    number of times. Returns a slope and intercept of line of best fit. """

    #   Convert to boolean image --- pixels are either 1 (white) or 0 (black)
    img = img_der(im, tuples = False).convert(mode="1")
    data = img.getdata()
    xs, ys = array_to_points(list(data), img.size)

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

        for k in range(pnum):
            x = xs[k]
            y = ys[k]
            d = dist_point_to_line((x, y), (m, b))
            if d <= threshold:
                dists.append(d)

        if len(dists) > max_good_points:
            max_good_points = len(dists)
            good_line = (m, b)

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


if __name__ == '__main__':
    img_num = 3068
    file = "Test/%s.0.png" % img_num
    file_train = "Test/%s.0-annotation.png" % img_num
    im = Image.open(file_train)
    im = im.convert(mode="L", dither=None)
    pyp_image = mpimg.imread(file)
    m, b = ransac(im)
    plt.imshow(pyp_image)
    plt.plot([0, 1280], [m*x + b for x in [0, 1280]], 'w', linewidth=4.0)
    plt.xlim([0, 1280])
    plt.ylim([0, 722])
    plt.show()
