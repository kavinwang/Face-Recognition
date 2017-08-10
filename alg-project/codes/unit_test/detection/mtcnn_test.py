from unit_test import test_utils
from detection import mtcnn

if __name__ == '__main__':
    test_utils.test(mtcnn, '../../../scripts/detection/mtcnn_script.txt')
