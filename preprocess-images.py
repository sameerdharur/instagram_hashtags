import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import numpy as np
from tqdm import tqdm

import config
import data
import utils
# from resnet import resnet as caffe_resnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet50(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_insta_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.InstaImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    cudnn.benchmark = True

    net = Net().to(device)
    net.eval()

    loader = create_insta_loader(config.train_path, config.val_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(config.preprocessed_path, libver='latest') as fd:
        # features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        # coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')
        features = np.zeros(features_shape)
        img_ids = []
        i = j = 0
        for ids, imgs in tqdm(loader):
            # imgs = Variable(imgs.cuda(async=True), volatile=True)
            # print(imgs.size())
            imgs = imgs.to(device)
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            img_ids.extend(ids)
            i = j


if __name__ == '__main__':
    main()
