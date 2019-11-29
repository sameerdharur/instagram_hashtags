import os
from os.path import expanduser
import platform

# paths
if 'debian' in platform.dist():
	qa_path = expanduser("~") + '/Captioning_Master/instagram_captioning'
else:
	qa_path = 'C:\Captioning_Master\instagram_captioning'  # directory containing the question and annotation jsons
train_path = os.path.join(qa_path, 'data', 'train')  # directory of training images
val_path = os.path.join(qa_path, 'data', 'val')  # directory of validation images
test_path = os.path.join(qa_path, 'data', 'test')  # directory of test images
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
captions_vocabulary_path = os.path.join(qa_path, 'cap_vocab.pkl')  # path where the used vocabularies for question and answers are saved to
hashtags_vocabulary_path = os.path.join(qa_path, 'hash_vocab.pkl')


task = 'OpenEnded'
dataset = 'instagram'

# preprocess config
preprocess_batch_size = 16
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 50
batch_size = 17
test_batch_size = 1
initial_lr = 1e-2  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000
