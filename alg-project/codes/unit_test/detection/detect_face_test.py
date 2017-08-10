from unit_test import test_utils
from detection import detect_face

if __name__ == '__main__':
    test_utils.test(detect_face, '../../../scripts/detection/detect_face_script.txt')
