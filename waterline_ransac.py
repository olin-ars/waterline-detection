#!/usr/bin/env python

#   Python libraries
pass

#   Outside libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

def array_to_points(arr):
    """ Takes a 2D numpy array of boolean values and returns a list of tuples
    for coordinates where True occurs. """

    points = []

    for y, row in enumerate(arr):
        #print(row)
        for x, val in enumerate(row):
            if val>=1:
                points.append((x, y))

    print(len(points))
    for item in points[:]:
        #plt.plot(item[1], item[0], '.')
        pass
    plt.show()

def ransac(im, points=5, trials=10):
    """ Runs ransac on an image, sampling 'points' number of points 'trials'
    number of times. Returns a slope and intercept of line of best fit. """

    img = img_der(im, tuples = False).convert(mode="1")
    data = np.array(img)
    array_to_points(data)

if __name__ == '__main__':
    im = Image.open("Test/2006.0-annotation.png")
    im = im.convert(mode="L")
    #im.show()
    ransac(im)
