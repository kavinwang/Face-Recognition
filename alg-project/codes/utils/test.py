import csv
from PIL import Image
import threadpool
import os
import cv2
import tensorflow as tf
from PIL import Image
import shutil


def read_from_batch3(path, src_path, cropped_dir):
    reader = csv.reader(open(path, 'r+', encoding='utf-8'))
    lines = [line for line in reader]
    work_queue = []
    for line in lines[1:]:
        if float(line[6]) < 80 or float(line[7]) < 80:
            continue
        coordination = tuple(map(int, [float(line[4]), float(line[5]), float(line[4]) + float(line[6]),
                                       float(line[5]) + float(line[7])]))
        work_queue.append(([src_path, line[1], coordination, cropped_dir], None))
    return work_queue


def img_crop(src_path, img_path, coordination, cropped_dir):
    path = os.path.join(src_path, img_path)
    img = Image.open(path)
    cropped = img.crop(coordination)
    output_dir = os.path.join(cropped_dir, img_path.split('/')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cropped.save(os.path.join(cropped_dir, img_path))


def check_num_of_channel(path):
    minimal = 100
    maximal = 0
    for classes in os.listdir(path):
        category = os.path.join(path, classes)
        for img_path in os.listdir(category):
            img = cv2.imread(os.path.join(category, img_path))
            minimal = min([img.shape[2], minimal])
            maximal = max([img.shape[2], maximal])
    return minimal, maximal


def img_crawler(input_dir, output_dir):
    work_queue = []
    classes = os.listdir(input_dir)
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls, 'DLKface')
        for img_path in os.listdir(cls_dir):
            if not os.path.exists(os.path.join(output_dir, cls)):
                os.mkdir(os.path.join(output_dir, cls))
            # img = Image.open(os.path.join(cls_dir, img_path))
            output_filename = os.path.join(output_dir, cls, img_path.split('_rfy')[0])
            # img.save(output_filename)
            work_queue.append(([os.path.join(cls_dir, img_path), output_filename], None))
            # shutil.copyfile(os.path.join(cls_dir, img_path), output_filename)
    return work_queue


def copy_img(input_path, output_path):
    img = Image.open(input_path)
    img.save(output_path)


if __name__ == '__main__':
    work_queue = []
    # path = '/run/media/csairmind/DataDevices/workdatas/Face/UMDfaces/src/umdfaces_batch2/umdfaces_batch2_ultraface.csv'
    # src_path = '/run/media/csairmind/DataDevices/workdatas/Face/UMDfaces/src/umdfaces_batch2'
    # cropped_dir = '/run/media/csairmind/DataDevices/workdatas/Face/UMDfaces/crop/umdfaces_batch2'
    input_dir = '/run/media/csairmind/DataDevices/workdatas/Face/WLFDB/WLFDB6025_src'
    output_dir = '/run/media/csairmind/DataDevices/workdatas/Face/WLFDB/WLFDB6025_crop'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    work_queue = img_crawler(input_dir, output_dir)
    # work_queue = read_from_batch3(path, src_path, cropped_dir)
    pool = threadpool.ThreadPool(200)
    requests = threadpool.makeRequests(copy_img, work_queue)
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print('mission complete!')
    # img_path = '/run/media/csairmind/DataDevices/workdatas/Face/ORL'
    # minimal, maximal = check_num_of_channel(img_path)
    # print("minimal: ", minimal, "maximal: ", maximal)
    # filename = '/run/media/csairmind/DataDevices/workdatas/Face/CSAIR/crop/180591/00c8c27a-7500-4826-a1d7-2ecfe6d010cf.png'
    # file_contents = tf.read_file(filename)
    # image = tf.image.decode_png(file_contents, channels=3)
    #
    # with tf.Session() as sess:
    #     # filename = '/run/media/csairmind/DataDevices/workdatas/Face/CSAIR/crop/180591/00c8c27a-7500-4826-a1d7-2ecfe6d010cf.png'
    #     # tf.local_variables_initializer().run()
    #     image = sess.run(image)
    #     img = Image.fromarray(image)
    #     img.save('/home/csairmind/Pictures/hello.png')
    # ls2 = []
    # ls1 = [1,2,3,4,5,6]
    # ls3 = [1, 2, 3, 4, 5, 6]
    # ls2.append([ls1, 7])
    # ls2.append([ls3, 9])
    # print(ls2)
