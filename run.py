from package.feature.data_processing import print_text_from_feature
from package.ml_training.train import print_something_from_train

if __name__ == '__main__':
    print_text_from_feature('Hello from feature')
    print_something_from_train('Hello from train')