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
from utils.contest_miou import compute_mean_ioU
from utils.transforms import transform_parsing, transform_image
from copy import deepcopy
import cv2
from inplace_abn import InPlaceABN
from IPython import embed
    
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

def valid(model, valloader, input_size, num_samples, save_flag=False):
    print('testing: {} samples'.format(num_samples))
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            images, meta = batch
            num_images = images.size(0)
            if (index+1) % 2 == 0:
                print('{} processd {} remained'.format(index * num_images, num_samples))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            h = meta['height'].numpy()
            w = meta['width'].numpy()

            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            outputs, _ = model(images.cuda())

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
                images = images.data.cpu().numpy()  # NCHW NHWC
                parsing_preds_bs = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                parsing_preds[idx:idx + num_images, :, :] = parsing_preds_bs
                if save_flag:
                    for i in range(len(meta['name'])):
                        save_dir = meta['name'][i].replace('image', 'pred').replace('.jpg', '.png')
                        if not os.path.exists(os.path.dirname(save_dir)):
                            os.makedirs(os.path.dirname(save_dir))
                        pred = transform_parsing(parsing_preds_bs[i], c[i], s[i], w[i], h[i], input_size)
                        pred = postprocess(pred)
                        cv2.imwrite(save_dir, pred)
                        
                        save_dir = meta['name'][i].replace('image', 'pred_plot')
                        if not os.path.exists(os.path.dirname(save_dir)):
                            os.makedirs(os.path.dirname(save_dir))
                        
                        img = transform_image(denorm(images[i]), c[i], s[i], w[i], h[i], input_size)
                        plot = vis_parsing_maps(img, pred)
                        cv2.imwrite(save_dir, plot)

                idx += num_images
    parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers

def postprocess(label_pr):
    
    eyebrow_index = np.argwhere((label_pr==2) | (label_pr==3))
    eye_index = np.argwhere((label_pr==4) | (label_pr==5))
    ear_index = np.argwhere((label_pr==13) | (label_pr==14))
    eyeshadow_index = np.argwhere((label_pr==11) | (label_pr==12))
    nose_index = np.argwhere((label_pr==6))
    face_index = np.argwhere((label_pr==1))
    if not nose_index.size==0: center_idex = int(nose_index.mean(axis=0)[1])
    elif not eye_index.size==0: center_idex = int(eye_index.mean(axis=0)[1])
    elif not face_index.size==0: center_idex = int(face_index.mean(axis=0)[1])
    else: center_idex = int(label_pr.shape[1]/2)

    for item_idex in eyebrow_index:
        label_pr[item_idex[0], item_idex[1]] = 3 if item_idex[1]<center_idex else 2
    for item_idex in eye_index:
        label_pr[item_idex[0], item_idex[1]] = 5 if item_idex[1]<center_idex else 4
    for item_idex in ear_index:
        label_pr[item_idex[0], item_idex[1]] = 14 if item_idex[1]<center_idex else 13
    for item_idex in eyeshadow_index:
        label_pr[item_idex[0], item_idex[1]] = 12 if item_idex[1]<center_idex else 11

    return label_pr

def vis_parsing_maps(im, img_pred):
    # Colors for all 20 parts
    part_colors = [[0,0,0],[255, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    vis_im = im.copy()
    img_pred_color = np.zeros((img_pred.shape[0], img_pred.shape[1], 3)) + 255
    num_of_class = np.max(img_pred)
    for pi in range(1, num_of_class + 1):
        img_pred_color[img_pred == pi] = part_colors[pi]
    img_pred_color = img_pred_color.astype(np.uint8)

    vis_im = cv2.addWeighted(vis_im, 0.8, img_pred_color, 0.2, 0)
    fig_arr = np.concatenate([im, img_pred_color, vis_im], axis=1)
    return fig_arr

def denorm(img):
    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    img = img.transpose((1,2,0))*(0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    img = np.uint8(255*img)
    return img

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

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, save_flag=True)

    # mIoU, f1 = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'val', reverse=False)

    # print(mIoU)
    # print(f1)

if __name__ == '__main__':
    main()

# python contest_evaluate.py --data-dir ../CVPR-PIC-DATA/ --restore-from ./snapshots/pic2021-05-06T15-54-00/epoch_19.pth --gpu 3 --batch-size 12 --input-size 473,473 --dataset test --num-classes 18
