import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
# from scipy.misc import imsave
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from networks.EAGR import EAGRNet
from collections import OrderedDict
from dataset.helen import HelenDataSet
from dataset.pic import PICDataSet
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy
import cv2
from inplace_abn import InPlaceABN

DATA_DIRECTORY = './datasets/Helen'
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="choose gpu numbers") 
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    return parser.parse_args()

def valid(model, valloader, input_size, num_samples, dir=None):
    print('testing: {} samples'.format(num_samples))
    model.eval()
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, edge, meta = batch
            num_images = image.size(0)
            if (index+1) % 2 == 0:
                print('{} processd {} remained'.format(index * num_images, num_samples))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs, _ = model(image.cuda())

            if isinstance(outputs, list):
                for output in outputs:
                    parsing = output
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

                    idx += nums
            else:
                parsing = outputs
                parsing = interp(parsing).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                if dir is not None:
                    for i in range(len(meta['name'])):
                        cv2.imwrite(os.path.join(dir, meta['name'][i] + '.png'), np.asarray(np.argmax(parsing, axis=3))[i])
                idx += num_images
    parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = EAGRNet(args.num_classes, InPlaceABN)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = PICDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    num_samples = len(dataset)
    
    valloader = data.DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from
    state_dict_old = torch.load(restore_from, map_location='cuda:0')
    model.load_state_dict(state_dict_old)
    
    model.cuda()
    model.eval()

    save_path =  os.path.join(args.data_dir, 'full')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, save_path)
    # mIoU, f1 = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'test', reverse=True)

    print(mIoU)
    print(f1)

if __name__ == '__main__':
    main()
