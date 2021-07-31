import argparse
import logging
import time
# select GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np
# pre-define model: https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet.model import EfficientNet
from utils.utils import get_loaders, normalize, inv_normalize

parser = argparse.ArgumentParser( description='PyTorch efficientnet model playground')
parser.add_argument('--resume', '-r',       action='store_true',              help='resume from checkpoint')
parser.add_argument('--prefix',             default='default',     type=str,   help='prefix for model checkpoints')
parser.add_argument('--seed',               default=33201701,      type=int,   help='random seed')

parser.add_argument('--batch_size', '-b',   default=120,           type=int,   help='mini-batch size (default: 120)')
parser.add_argument('--epochs', '-e',       default=50 ,           type=int,   help='number of total epochs to run')
parser.add_argument('--lr',                 default=0.001,         type=float, help='learning rate for optimizer')
parser.add_argument('--image-size', '--is', default=224,           type=int,   help='resize input image (default: 224 for ImageNet)')
parser.add_argument('--data-directory',     default='../Restricted_ImageNet',type=str,   help='dataset inputs root directory')

parser.add_argument('--opt-level', '-o',    default='O1',          type=str,   help='Nvidia apex optimation level (default: O1)')
parser.add_argument('--model-name', '-m',   default='resnet18', type=str, help='Specify the varient of the model ')
args = parser.parse_args()

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader = get_loaders(args.data_directory, args.batch_size, \
                                            args.image_size, augment=True)
    # Load model and optimizer
    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=10).to(device)
    elif args.model_name == 'resnet34':
        model = models.resnet34(pretrained=False, num_classes=10).to(device)
    elif args.model_name == 'resnet50':
        model = models.resnet50(pretrained=False, num_classes=10).to(device)
    else:
        print('The model is not supported')
        return
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                            # momentum=args.momentum,
                            # weight_decay=args.weight_decay
                            )
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, verbosity=0)

        checkpoint = torch.load('./checkpoint/' + args.prefix + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        amp.load_state_dict(checkpoint['amp_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model..')
        epoch_start = 0
        prev_acc = 0.0
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, verbosity=0)
    criterion = nn.CrossEntropyLoss().to(device)

    # Logger
    result_folder = './logs/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logger = logging.getLogger(__name__)
    logname = args.prefix + '_' + args.opt_level + '.log'
    logfile = os.path.join(result_folder, logname)
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile
    )
    logger.info(args)

    # Training
    def train(epoch):
        print('\nEpoch: {:04}'.format(epoch))
        train_loss, correct, total = 0, 0, 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output_logit = model(normalize(data))
            loss = criterion(output_logit, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()
            preds = F.softmax(output_logit, dim=1)
            preds_top_p, preds_top_class = preds.topk(1, dim=1)

            train_loss += loss.item() * target.size(0)
            total += target.size(0)
            correct += (preds_top_class.view(target.shape) == target).sum().item()
            # if batch_idx > 150:
            #     break

        return (train_loss / batch_idx, 100. * correct / total)

    # Test
    def test(epoch):
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
    
                output_logit = model(normalize(data))
                loss = criterion(output_logit, target)
                preds = F.softmax(output_logit, dim=1)
                preds_top_p, preds_top_class = preds.topk(1, dim=1)
    
                test_loss += loss.item() * target.size(0)
                total += target.size(0)
                correct += (preds_top_class.view(target.shape) == target).sum().item()
                # if batch_idx > 150:
                #     break
        
        return (test_loss / batch_idx, 100. * correct / total)
            
    # Save checkpoint
    def checkpoint(acc, epoch):
        print('==> Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/' + args.prefix + '.pth'
        torch.save({
            'epoch': epoch,
            'acc': acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'amp_state_dict': amp.state_dict(),
            'rng_state': torch.get_rng_state(),
            }, save_path)
    
    # Run
    logger.info('Epoch  Seconds    Train Loss  Train Acc    Test Loss  Test Acc')
    start_train_time = time.time()
    for epoch in range(epoch_start, args.epochs):
        start_epoch_time = time.time()
        
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        epoch_time = time.time()
        logger.info('%5d  %7.1f    %10.4f  %9.4f    %9.4f  %8.4f',
            epoch, epoch_time - start_epoch_time, train_loss, train_acc, test_loss, test_acc)
        # Save checkpoint.
        if train_acc - prev_acc  > 0.1:
            prev_acc = train_acc
            checkpoint(train_acc, epoch)
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)


if __name__ == "__main__":
    main()