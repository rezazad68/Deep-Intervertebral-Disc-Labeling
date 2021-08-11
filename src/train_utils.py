# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Revised by Reza Azad (several functions added)

from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
# from Data2array import *
import matplotlib.pyplot as plt
import PIL
from transform_spe import *
import skimage
import cv2
from scipy import signal
import torch 



# normalize Image
def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi))


# Useful function to generate a Gaussian Function on given coordinates. Used to generate groudtruth.
def label2MaskMap_GT(data, shape, c_dx=0, c_dy=0, radius=5, normalize=False):
    """
    Generate a Mask map from the coordenates
    :param shape: dimension of output
    :param data : input image
    :param radius: is the radius of the gaussian function
    :param normalize : bool for normalization.
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M, N) = (shape[2], shape[1])
    if len(data) <= 2:
        # Output coordinates are reduced during post processing which poses a problem
        data = [0, data[0], data[1]]
    maskMap = []

    x, y = data[2], data[1]

    # Correct the labels
    x += c_dx
    y += c_dy

    X = np.linspace(0, M - 1, M)
    Y = np.linspace(0, N - 1, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Mean vector and covariance matrix
    mu = np.array([x, y])
    Sigma = np.array([[radius, 0], [0, radius]])

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Normalization
    if normalize:
        Z *= (1 / np.max(Z))
    else:
        # 8bit image values (the loss go to inf+)
        Z *= (1 / np.max(Z))
        Z = np.asarray(Z * 255, dtype=np.uint8)

    maskMap.append(Z)

    if len(maskMap) == 1:
        maskMap = maskMap[0]

    return np.asarray(maskMap)


def extract_all(list_coord_label, shape_im=(1, 150, 200)):
    """
    Create groundtruth by creating gaussian Function for every ground truth points for a single image
    :param list_coord_label: list of ground truth coordinates
    :param shape_im: shape of output image with zero padding
    :return: a 2d heatmap image.
    """
    shape_tmp = (1, shape_im[0], shape_im[1])
    final = np.zeros(shape_tmp)
    for x in list_coord_label:
        train_lbs_tmp_mask = label2MaskMap_GT(x, shape_tmp)
        for w in range(shape_im[0]):
            for h in range(shape_im[1]):
                final[0, w, h] = max(final[0, w, h], train_lbs_tmp_mask[w, h])
    return (final)


def extract_groundtruth_heatmap(DataSet):
    """
    Loop across images to create the dataset of groundtruth and images to input for training
    :param DataSet: An array containing [images, GT corrdinates]
    :return: an array containing [image, heatmap]
    """
    [train_ds_img, train_ds_label] = DataSet

    tmp_train_labels = [0 for i in range(len(train_ds_label))]
    tmp_train_img = [0 for i in range(len(train_ds_label))]
    train_ds_img = np.array(train_ds_img)

    for i in range(len(train_ds_label)):
        final = extract_all(train_ds_label[i], shape_im=train_ds_img[0].shape)
        tmp_train_labels[i] = normalize(final[0, :, :])

    tmp_train_labels = np.array(tmp_train_labels)

    for i in range(len(train_ds_img)):
        print(train_ds_img[i].shape)
        tmp_train_img[i] = (normalize(train_ds_img[i][:, :, 0]))

    tmp_train_labels = np.expand_dims(tmp_train_labels, axis=-1)
    tmp_train_img = np.expand_dims(train_ds_img, axis=-1)
    return [tmp_train_img, tmp_train_labels]


class image_Dataset(Dataset):
    def __init__(self, image_paths, target_paths, use_flip = True):  # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.num_vis_joints = []
        self.use_flip = use_flip

    @staticmethod
    def rotate_img(img):
        img = np.rot90(img)
        #img = np.flip(img, axis=0)
        img = np.flip(img, axis=1)
        return img

    
    def get_posedata(self, img, msk, num_ch=11):
        msk = msk[:, :, 0]
        msk = self.rotate_img(msk)
        #img = self.rotate_img(img)
        ys = msk.shape
        ys_ch = np.zeros([ys[0], ys[1], num_ch])
        msk_uint = np.uint8(np.where(msk >0.2, 1, 0))
        
        num_labels, labels_im = cv2.connectedComponents(msk_uint)
        self.num_vis_joints.append(num_labels-1)
        try:
            # the <0> label is the background
            for i in range(1, num_labels):
                y_i = msk * np.where(labels_im == i, 1, 0)
                ys_ch[:,:, i-1] = y_i
        except:
            print(num_labels)
            # plt.imshow(img, cmap='gray')
            # plt.savefig('spine/img.png')
            # plt.imshow(msk, cmap='gray')
            # plt.savefig('spine/mks.png')
            # plt.imshow(msk_uint, cmap='gray')
            # plt.savefig('spine/msk_uint.png')
            
        ys_ch = np.rot90(ys_ch)
        ys_ch = np.flip(ys_ch, axis=1)
        vis = np.zeros((num_ch, 1))
        vis[:num_labels-1] = 1
        return img, ys_ch, vis

    @staticmethod
    def bluring2D(data, kernel_halfsize=3, sigma=1.0):
        x = np.arange(-kernel_halfsize,kernel_halfsize+1,1)
        y = np.arange(-kernel_halfsize,kernel_halfsize+1,1)
        xx, yy = np.meshgrid(x,y)
        kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
        filtered = signal.convolve(data, kernel, mode="same")
        return filtered


    def transform(self, image, mask):
        #print(image.shape)
        image = normalize(image[:, :, 0])
        #image = skimage.exposure.equalize_adapthist(image, kernel_size=10, clip_limit=0.02)
        image = np.expand_dims(image, -1)

        ## extract joints for pose model
        
        # Random horizontal flipping
        if self.use_flip:
            image, mask = RandomHorizontalFlip()(image, mask)

        # Random vertical flipping
        # image,mask = RandomVerticalFlip()(image,mask)
        # random90 flipping
        temp_img = np.zeros((image.shape[0], image.shape[1], 3))
        temp_img[:,:,0:1]= image
        temp_img[:,:,1:2]= image
        temp_img[:,:,2:3]= image
        image = temp_img
        # image,mask = RandomangleFlip()(image,mask)

        # Transform to tensor
        
        image, mask = ToTensor()(image, mask)
        
        return image, mask

    def __getitem__(self, index):
        mask = self.target_paths[index]
        # maxval = np.max(np.max(mask))
        # mask = np.where(mask==maxval, 1, 0)
        # mask = self.bluring2D(mask[:,:,0], kernel_halfsize=15, sigma=3)
        # mask /= np.max(np.max(mask))
        # plt.imshow(mask, cmap='gray')
        # plt.savefig('spine/mask.png')
        
        mask = cv2.resize(mask, (256, 256))
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis= -1)

        image = self.image_paths[index]
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis= -1)
        
        image, mask, vis  = self.get_posedata(image, mask, num_ch=11)
        t_image, t_mask = self.transform(image, mask)
        
        vis = torch.FloatTensor(vis)
        return t_image, t_mask, vis

    def __len__(self):  # return count of sample we have
        
        return len(self.image_paths)



class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize


def save_epoch_res_as_image(inputs, targets, epoch_num, flag_gt):

    targets = targets.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()

    if not(flag_gt):
       targets[np.where(targets<0.5)] = 0 

    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0, 0], dtype=np.uint8)
    for y, x in zip(targets, inputs):
        y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
        y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
        for ych, hue_i in zip(y, hues):
            ych = ych/np.max(np.max(ych))
            ych[np.where(ych<0.5)] = 0
            # ych = cv2.GaussianBlur(ych,(15,15),cv2.BORDER_DEFAULT)

            ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
            ych = np.uint8(255*ych/np.max(ych))
            
            colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
            colored_ych[:, :, 0] = ych_hue
            colored_ych[:, :, 1] = blank_ch
            colored_ych[:, :, 2] = ych
            colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

            y_colored += colored_y
            y_all += ych

        x = np.moveaxis(x, 0, -1)
        x = x/np.max(x)*255

        x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
        for i in range(3):
            x_3ch[:, :, i] = x[:, :, 0]
        
        img_mix = np.uint8(x_3ch*0.5 + y_colored*0.5)
        
        txt = ''
        if flag_gt:
            txt = f'visualize/epo_{epoch_num:3d}_gt.png'
        else:
            txt = f'visualize/epo_{epoch_num:3d}_pr.png'

        cv2.imwrite(txt, img_mix)
        break


from torchvision.utils import make_grid
def save_epoch_res_as_image2(inputs, outputs, targets, epoch_num, target_th=0.4, pretext=False):
    max_epoch = 500
    target_th = target_th + (epoch_num/max_epoch*0.2)
    targets = targets.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()

    clr_vis_Y = []

    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0, 0], dtype=np.uint8)

    for Y in [targets, outputs]:
        for y, x in zip(Y, inputs):
            y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
            y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
            for ych, hue_i in zip(y, hues):
                ych = ych/np.max(np.max(ych))
                ych[np.where(ych<target_th)] = 0
                # ych = cv2.GaussianBlur(ych,(15,15),cv2.BORDER_DEFAULT)

                ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
                ych = np.uint8(255*ych/np.max(ych))
                
                colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
                colored_ych[:, :, 0] = ych_hue
                colored_ych[:, :, 1] = blank_ch
                colored_ych[:, :, 2] = ych
                colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

                y_colored += colored_y
                y_all += ych

            x = np.moveaxis(x, 0, -1)
            x = x/np.max(x)*255

            x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
            for i in range(3):
                x_3ch[:, :, i] = x[:, :, 0]
            
            img_mix = np.uint8(x_3ch*0.5 + y_colored*0.5)
            # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            clr_vis_Y.append(img_mix)
            
    
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    if pretext:
        txt = f'visualize/{epoch_num:0=4d}_test_result.png'
    else: 
        txt = f'visualize/t2/epoch_{epoch_num:0=4d}_res2.png'
    res = np.transpose(trgts.numpy(), (1,2,0))
    print(res.shape)
    cv2.imwrite(txt, res)

    # plt.imshow()
    # plt.savefig(txt)
    # cv2.imwrite(txt, )




def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N
    





class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

import math
def sigmoid(x):
    x = np.array(x)
    x = 1/(1+np.exp(-x))
    x[x<0.0] = 0
    return x
    # return np.where(1/(1+np.exp(-x))>0.5, 1., 0.)

import copy
def save_attention(inputs, outputs, targets, att, target_th=0.5):
    targets = targets.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    inputs  = inputs.data.cpu().numpy()
    att     = att.detach().to('cpu')
    
    att = torch.sigmoid(att).numpy()
    att = np.uint8(att*255)
    att[att<128+80] = 0
    # att     = sigmoid(att)
    att     = cv2.resize(att, (256, 256))
    att     = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    rgbatt  = copy.copy(inputs[0])
    rgbatt  = np.moveaxis(rgbatt, 0, -1)
    rgbatt = rgbatt*255*0.5+ att*0.5
    # at2[:, :, 0] = att 
    # at2[:, :, 1] = att 
    # at2[:, :, 2] = att 
    # at2     = cv2.applyColorMap(np.uint8(at2*255), cv2.COLORMAP_JET)
    # rgbatt = at2
    # rgbatt  = cv2.addWeighted(rgbatt, 0.7, att, 0.3, 0)

    clr_vis_Y = []

    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0, 0], dtype=np.uint8)

    for Y in [targets, outputs]:
        for y, x in zip(Y, inputs):
            y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
            y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
            for ych, hue_i in zip(y, hues):
                ych = ych/np.max(np.max(ych))
                ych[np.where(ych<target_th)] = 0
                # ych = cv2.GaussianBlur(ych,(15,15),cv2.BORDER_DEFAULT)

                ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
                ych = np.uint8(255*ych/np.max(ych))
                
                colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
                colored_ych[:, :, 0] = ych_hue
                colored_ych[:, :, 1] = blank_ch
                colored_ych[:, :, 2] = ych
                colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

                y_colored += colored_y
                y_all += ych

            x = np.moveaxis(x, 0, -1)
            x = x/np.max(x)*255

            x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
            for i in range(3):
                x_3ch[:, :, i] = x[:, :, 0]
            
            img_mix = np.uint8(x_3ch*0.5 + y_colored*0.5)
            # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            clr_vis_Y.append(img_mix)
            
    clr_vis_Y.append(rgbatt)
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)
    txt = './visualize/attention_visualization.png'
    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(txt, res)

    # plt.imshow()
    # plt.savefig(txt)
    # cv2.imwrite(txt, )

