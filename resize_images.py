
# TODO use argparse ( https://stackoverflow.com/questions/7427101/simple-argparse-example-wanted-1-argument-3-results )

import os.path, sys
import shutil
from imageio import imread as imread
from imageio import imsave as imsave

from scipy.misc import imread as imread_scipy
from skimage.color import rgb2gray

import numpy as np
import cv2
import scipy.ndimage


IMG_FILE_TYPES = [".jpg", ".jpeg", ".png"]

class resize_images():

    # def __init__(self, src, dst, dsize, scale_percent=None):
    #
    #     self.src = src
    #     self.dst = dst
    #     self.dsize = int(dsize)
    #     self.scale_percent = int(scale_percent) if scale_percent is not None else None

    def __init__(self, src, dst, scale_percent, dsize_h, dsize_w, allow_for_disproportion=None):
        """
        Class for resizing of all images in a given root directory to a destination, while keeping the directory tree structure.
        Note that at least one of scale_percent, dsize_h, dsize_w has to be non-zero.
        :param src: source dire
        :param dst: destination dir
        :param scale_percent: int between 1 and 100. if 0 --> ignored (and then resizing will be done with dsize)
        :param dsize_h: int, new height. if 0 --> ignored
        :param dsize_w: int, new width. if 0 --> ignored
        :param allow_for_disproportion: 0 or 1, if 1 it allows for free resizing of images. if None, it defaults to 0.
        """

        # parse arguments
        self.src = src
        self.dst = dst
        self.scale_percent, self.dsize_h, self.dsize_w = int(scale_percent), int(dsize_h), int(dsize_w)
        self.allow_for_disproportion = 0 if allow_for_disproportion is None else int(allow_for_disproportion)

        # first, copy the directory structure to self.dst
        self.copy_dir_structure()

        # now, do the rescaling
        self.resize_all()

    def copy_dir_structure(self, src=None, dst=None):

        print("copying dir structure")

        if src is None: src = self.src
        if dst is None: dst = self.dst

        def ig_f(dir, files):
            return [f for f in files if os.path.isfile(os.path.join(dir, f))]

        shutil.copytree(src, dst, ignore=ig_f)


    def resize_all(self, src=None, dst=None):

        print("*** started resizing ***")

        if src is None: src = self.src
        if dst is None: dst = self.dst

        # keep track of progress and some statistics
        curr_dir, curr_file_i, img_counter = "", 1, 0
        # total_dirs = len(next(os.walk(src))[1]) + 1 # this doesn't count subdirectories recursivly todo total_dirs
        _root, _dirs, _files = os.walk(src).__next__()
        total_files = len(_files)
        smallest_dim, biggest_dim = np.inf, 0

        for subdir, dirs, files in os.walk(src):
            for file in files:

                # print progress
                print("reached file %s / %s" % (str(curr_file_i), str(total_files)), end="\r", flush=True)
                # print("reached file %s %s / %s" % (file, str(curr_file_i), str(total_files)), end="\n", flush=True)
                curr_file_i += 1
                # if subdir != curr_dir:
                #     curr_dir = subdir
                #     curr_dir_i += 1
                #     print("current dir: ", curr_dir)
                #     # print(curr_dir_i, "/", total_dirs, " current dir: ", curr_dir) # same todo total_dirs

                try:
                    file_path = os.path.join(subdir, file)
                    if self.file_is_image(file_path):
                        if self.filter_image(file_path):
                            dst_path = dst + subdir.split(src)[1] + os.sep + file
                            result_dims = self.resize_img(file_path, dst_path)
                            for d in result_dims:
                                smallest_dim = d if d < smallest_dim else smallest_dim
                                biggest_dim = d if d > biggest_dim else biggest_dim
                            img_counter += 1
                except:
                    pass

        print("*** finished resizing ***")
        print("total number of images resized: ", img_counter)
        print("smallest dimension in output: %i biggest: %i" % (smallest_dim, biggest_dim))

    def file_is_image(self, file_path):

        if not os.path.isfile(file_path):
            return False

        is_img = False
        for img_file_type in IMG_FILE_TYPES:
            if file_path.endswith(img_file_type):
                is_img = True

        return is_img

    def filter_image(self, img_path):
        """
        returns True if we want this image, False if not
        """

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # filter by size : no enlarging of images is allowed   # todo make this optional
        if img.shape[0] < self.dsize_h or img.shape[1] < self.dsize_w:
            return False

        if img.shape[0]!=img.shape[1] and not self.allow_for_disproportion:
            return False

        return True


    def resize_img(self, img_path, dst_path):

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # figure new dimensions
        dim = self.calc_dim(img.shape)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, resized)

        return dim


    def calc_dim(self, img_shape):

        dim = None

        if self.dsize_h != 0 or self.dsize_w != 0:  # in this case we use dsize
            if self.dsize_h == 0 or self.dsize_w == 0:  # only one dim was given
                dsize = max(self.dsize_h, self.dsize_w)
                if (img_shape[0] == img_shape[1]):
                    dim = (dsize, dsize)
                else:
                    # if the image is not square and only one dim was given, this should be the longest
                    old_max = max(img_shape)
                    scaling_factor = dsize / old_max
                    width = int(img_shape[1] * scaling_factor)
                    height = int(img_shape[0] * scaling_factor)
                    dim = (width, height)
            else: # two dims were given
                if self.allow_for_disproportion:
                    dim = (self.dsize_w, self.dsize_h)
                else:
                    scaling_factor = min((self.dsize_h / img_shape[0]), (self.dsize_w / img_shape[1]))
                    width = int(img_shape[1] * scaling_factor)
                    height = int(img_shape[0] * scaling_factor)
                    dim = (width, height)
        else: # in this case we use scale_percent
            width = int(img_shape[1] * self.scale_percent / 100)
            height = int(img_shape[0] * self.scale_percent / 100)
            dim = (width, height)

        return dim



def main():

    if len(sys.argv) < 6:
        raise Exception("missing input (required: src dst scale_percent dsize_h dsize_w (optional: allow_for_disproportion)")

    # prepare a valid destination dir
    dst_dir = sys.argv[2]
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)

    my_resizer = resize_images(*sys.argv[1:])


if __name__ == '__main__':

    main()
