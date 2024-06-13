import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
# from data_loader import GetLoader
from torchvision import datasets


import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
# from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
import torch.cuda
# from model import CNNModel
# from test import test

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU')

    source_dataset_name = 'SOURCE'
    target_dataset_name = 'TARGET'
    # source_image_root = os.path.join('dataset', source_dataset_name)
    # target_image_root = os.path.join('dataset', target_dataset_name)
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 16
    image_size = 28
    n_epoch = 50

    manual_seed = random.randint(1, 20000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # load data

    # img_transform_source = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])

    # img_transform_target = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # dataset_source = datasets.MNIST(
    # #     root='dataset',
    #     train=True,
    # #     transform=img_transform_source,
    # #     download=True
    # )



    # train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    dataset_source = GetLoader(
    #     data_root=os.path.join(target_image_root, 'mnist_m_train'),
    #     images=train_x,
        data_list=np.argmax(train_chunk_y, axis=1),
        test=False
    #     transform=img_transform_target
    )
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    dataset_target = GetLoader(
    #     data_root=os.path.join(target_image_root, 'mnist_m_train'),
    #     images=test_x,
        data_list=np.argmax(test_chunk_y, axis=1),
        test=True
    #     transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)

    # load model

    my_net = CNNModel()

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    # if cuda:
    #     my_net = my_net.cuda()
    #     loss_class = loss_class.cuda()
    #     loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = next(data_source_iter)
    #         dataiter = iter(data_source)
    #         data = dataiter.next()
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()
            
    #         s_label = s_label.type(torch.LongTensor)
    #         domain_label = s_label
    #         if cuda:
    #             s_img = s_img.cuda()
    #             s_label = s_label.cuda()
    #             domain_label = domain_label.cuda()

    #         print(s_img.shape)
            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            err_s_domain = loss_domain(domain_output, domain_label)
            err_s_label = loss_class(class_output, s_label)

            # training model using target data
            data_target = next(data_target_iter)
    #         t_img, _ = data_target
            t_img, t_label = data_target
            t_label = t_label.type(torch.LongTensor)
            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()
    #         domain_label = t_label
    #         domain_label = torch.where(t_label == 1, domain_label, 13)
    #         domain_label = torch.where(t_label == 2, domain_label, 14)
    #         domain_label = torch.where(t_label == 3, domain_label, 15)
    #         domain_label = torch.where(t_label == 4, domain_label, 16)
    #         domain_label = torch.where(t_label == 5, domain_label, 17)
    #         domain_label = torch.where(t_label == 6, domain_label, 18)
    #         domain_label = torch.where(t_label == 7, domain_label, 19)
    #         domain_label = torch.where(t_label == 8, domain_label, 20)
    #         domain_label = torch.where(t_label == 9, domain_label, 21)
    #         domain_label = torch.where(t_label == 10, domain_label, 22)
    #         domain_label = torch.where(t_label == 11, domain_label, 23)

    #         if cuda:
    #             t_img = t_img.cuda()
    #             domain_label = domain_label.cuda()
    #             s_label = s_label.cuda()
        

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
    #         err = err_s_label
            err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                    err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
    #         torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))
            torch.save(my_net, os.path.join('/home/huang/LIBS/', 'libs_model_epoch_current.pth'))

        print('\n')
    #     accu_s, accu_s_dm = test('train')
        accu_s = test('train')
        print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
    #     print('Domain Accuracy of the %s dataset: %f' % ('mnist', accu_s_dm))
    #     accu_t, accu_t_dm = test('test')
        accu_t = test('test')
        print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
    #     print('Domain Accuracy of the %s dataset: %f' % ('mnist', accu_t_dm))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
    #         torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))
    #         torch.save(my_net,os.path.join('dataset/models', 'libs_model_epoch_best.pth'))

    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/libs_model_epoch_best.pth')