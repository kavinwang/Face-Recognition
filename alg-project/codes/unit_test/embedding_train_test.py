from unit_test import test_utils
from embedding import embedding_train

if __name__ == '__main__':
    test_utils.test(embedding_train, '../../scripts/embedding_train_script.txt')
