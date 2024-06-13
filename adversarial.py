import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
# from data_loader import GetLoader
from torchvision import datasets


def test(dataset_name):
#     assert dataset_name in ['MNIST', 'mnist_m']

#     model_root = 'models'
#     image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 16
#     image_size = 28
    alpha = 0

    """load data"""

#     img_transform_source = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.1307,), std=(0.3081,))
#     ])

#     img_transform_target = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])

    if dataset_name == 'test':
#         print('test --')
#         test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(
    #     data_root=os.path.join(target_image_root, 'mnist_m_train'),
    #     images=test_x,
        data_list=np.argmax(test_chunk_y, axis=1),
        test=True
    #     transform=img_transform_target
        )
#         domain_label = torch.ones(batch_size).long()
    else:
        dataset = GetLoader(
    #     data_root=os.path.join(target_image_root, 'mnist_m_train'),
    #     images=test_x,
        data_list=np.argmax(train_chunk_y, axis=1),
        test=False
    #     transform=img_transform_target
        )
#         domain_label = torch.zeros(batch_size).long()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    

    

    """ test """

    my_net = torch.load(os.path.join('/home/huang/LIBS/', 'libs_model_epoch_current.pth'))
    my_net = my_net.eval()

#     if cuda:
#         my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    n_correct_dm = 0
#     n_total_dm = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

#         if cuda:
#             t_img = t_img.cuda()
#             t_label = t_label.cuda()
#             domain_label = domain_label.cuda()
            
        class_output, domain_output = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
#         pred_dm = domain_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
#         n_correct_dm += pred_dm.eq(domain_label[:batch_size].data.view_as(pred_dm)).cpu().sum()
#         print(pred)
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
#     accu_dm = n_correct_dm.data.numpy() * 1.0 / n_total

#     return accu, accu_dm
    return accu

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=3))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 128, kernel_size=3))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(128))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_conv3', nn.Conv2d(128, 256, kernel_size=3))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(256))
#         self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool3', nn.MaxPool2d(2))
        self.feature.add_module('f_relu3', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
#         self.class_classifier.add_module('c_fc1', nn.Linear(50 * 41 * 53, 100))
        self.class_classifier.add_module('c_fc1', nn.Linear(256 * 2 * 6, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
#         self.class_classifier.add_module('c_drop1', nn.Dropout())
#         self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
#         self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
#         self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 5))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
#         self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 41 * 53, 100))
#         self.domain_classifier.add_module('d_fc1', nn.Linear((256 * 4 * 12, 100))
#         self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
#         self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(256 * 2 * 6, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.type(torch.float32)
        input_data = input_data.expand(input_data.data.shape[0], 1,32, 64)
        feature = self.feature(input_data)
#         print(feature.shape)
#         feature = feature.view(-1, 50 * 41 * 53)
        feature = feature.view(-1, 256 * 2 * 6)
#         print(feature.shape)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
#         print(reverse_feature.shape)
        class_output = self.class_classifier(feature)
#         print(class_output.shape)
        domain_output = self.domain_classifier(reverse_feature)
#         print(domain_output.shape)

        return class_output, domain_output

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_list, transform=None, test=False):
#         self.root = data_root
        self.transform = transform
        self.test = test
        self.data_list = data_list
#         print(str(self.data_list))
#         f = open(data_list, 'r')
#         data_list = f.readlines()
#         f.close()

        self.n_data = data_list.shape[0]


    def __getitem__(self, item):
        if (self.test ):
#             print("test")
            images = test_chunk_x
        else:
            images = train_chunk_x
        imgs, labels = images[item], self.data_list[item]
#         imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data