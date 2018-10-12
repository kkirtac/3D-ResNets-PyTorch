from __future__ import print_function, division
import os
import sys
import glob
import errno
from shutil import copyfile
import time

def sample_frames(src_root_dirpath, dst_root_dirpath, sampling_rate=25):

    for dir_name in sorted(os.listdir(src_root_dirpath)):
        print(dir_name)
        filespath = os.path.join(os.path.abspath(src_root_dirpath), dir_name, '*.jpg')
        files = sorted(glob.glob(filespath))
        files_ = files[0:-1:sampling_rate]
        for f in files_:
            dst_dirpath = os.path.join(dst_root_dirpath, os.path.basename(os.path.dirname(f)))
            try:
                os.makedirs(dst_dirpath)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # raises the error again

            copyfile(f, os.path.join(dst_dirpath, os.path.basename(f)))


def count_jpg(dir_path):
    fh = open('count_jpg.txt', 'w')

    for dir_name in os.listdir(dir_path):
        filespath = os.path.join(os.path.abspath(dir_path), dir_name, '*.jpg')
        files = sorted(glob.glob(filespath))
        print('{} : {}'.format(dir_name, len(files)), file=fh)

    fh.close()



if __name__=="__main__":
    dir_path = sys.argv[1]
    dst_path = sys.argv[2]

    sample_frames(dir_path, dst_path)
