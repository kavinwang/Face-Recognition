import os
import random
import shutil
import re

inputfilelists = 'lfw:/home/csairmind/Desktop/test_input/lfw,umd:/home/csairmind/Desktop/test_input/umd'
outputdir = "/home/csairmind/Desktop/test_input"


def find_all_images(path, dataset, src_folder):
    classes = os.listdir(path)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
            dataset.append(ImageClass(class_name, image_paths, src_folder))


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths, src_folder):
        self.name = name
        self.image_paths = image_paths
        self.src_folder = src_folder

    def __str__(self):
        return self.src_folder + ': ' + self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

    def getsrcfolder(self):
        return self.src_folder


def get_dataset(paths):
    dataset = []
    print(paths)  # D:/CSAIR_FaceRecognize_v2/dataset/
    for path in paths.split(','):
        sign, path = path.split(':')
        if "lfw" == sign.lower():
            # path_exp = os.path.expanduser(path)
            # Here is the probleme
            # print(path_exp) # D ???
            find_all_images(path, dataset, sign.lower())
        elif "umd" == sign.lower():
            batches = os.listdir(path)
            batches.sort()
            for i in range(len(batches)):
                batchdir = os.path.join(path, batches[i])
                if os.path.isdir(batchdir):
                    find_all_images(batchdir, dataset, sign.lower())
        elif "abe" == sign.lower():
            class_name = ""
            image_paths = []
            images = os.listdir(path)
            images.sort()
            for image in images:
                if len(image_paths) == 0:
                    class_name = re.findall('[a-zA-Z]+', image.split(".")[0])
                if re.findall('[a-zA-Z]+', image.split(".")[0]) != class_name:
                    dataset.append(ImageClass(class_name[0], image_paths))
                    class_name = re.findall('[a-zA-Z]+', image.split(".")[0])
                    image_paths = []
                image_paths.append(os.path.join(path, image))
            dataset.append(ImageClass(class_name[0], image_paths, sign.lower()))

    return dataset


def find_random_class():
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    count = 0
    for filelist in inputfilelists:
        class_names = os.listdir(filelist)
        for name in class_names:
            if random.random() > 0.99:
                shutil.copytree(os.path.join(filelist, name), os.path.join(outputdir, name))
                count += 1
                if count >= 100:
                    break


def generate_validtion_set(paths, output_dir):
    filename = os.path.join(output_dir, "cross_combination.txt")
    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, "w+")
    dataset = []
    dataset = get_dataset(paths)
    num_of_classes = len(dataset)
    for i in range(num_of_classes):
        for j in range(i, num_of_classes):
            flag = 0
            if i == j:
                flag = 1
            print(random.choice(dataset[i].image_paths).split(output_dir)[1],
                  random.choice(dataset[j].image_paths).split(output_dir)[1], flag, file=f)
    f.close()


generate_validtion_set(inputfilelists, outputdir)
