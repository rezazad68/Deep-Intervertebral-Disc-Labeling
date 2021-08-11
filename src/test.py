## Original code from Neuropoly, Lucas
## Code modified by Reza Azad
from __future__ import print_function, absolute_import
import _init_paths
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import argparse
from Metrics import *
import torch
import numpy as np
from train_utils import *
# from Data2array import *
from pose_code.hourglass import hg
from pose_code.atthourglass import atthg
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import skimage
import pickle
from torch.utils.data import DataLoader 
import cv2
from sklearn.utils.extmath import cartesian
from skimage.feature import peak_local_max


## Functions from neuropoly
def retrieves_gt_coord(ds):
    coord_retrieved = []
    for i in range(len(ds[1])):
        coord_tmp = [[], []]
        for j in range(len(ds[1][i])):
            if ds[1][i][j][3] == 1 or ds[1][i][j][3] > 30:
                print('remove' + str(ds[1][i][j][3]))
                pass
            else:
                coord_tmp[0].append(ds[1][i][j][2])
                coord_tmp[1].append(ds[1][i][j][1])
        coord_retrieved.append(coord_tmp)
    return (coord_retrieved)
    
def prediction_coordinates(final, coord_gt):
    num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(np.uint8(np.where(final>0, 255, 0)))
    #centers = peak_local_max(final, min_distance=5, threshold_rel=0.3)

    centers = centers[1:] #0 for background
    coordinates = []
    for x in centers:
        coordinates.append([x[0], x[1]])
    #print('calculating metrics on image')
    mesure_err_disc(coord_gt, coordinates, distance_l2)
    mesure_err_z(coord_gt, coordinates, zdis)
    fp = Faux_pos(coord_gt, coordinates, tot)
    fn = Faux_neg(coord_gt, coordinates)
    faux_pos.append(fp)
    faux_neg.append(fn)
    
    #print(f'distance L2 is equal to {distance_l2}')
    #print(f'distance z is equal to {zdis}')
    #print(f'number of false positive is equal to {fp}')
    #print(f'number of false negative is equal to {fn}')
   
    
# main script
def main(args):
    global cuda_available
    cuda_available = torch.cuda.is_available()
    print('load image')
    # put image into an array
    with open(f'{args.datapath}_{args.modality}_ds',   'rb') as file_pi:       
         ds = pickle.load(file_pi)
    with open(f'{args.datapath}_{args.modality}_full', 'rb') as file_pi:
         full = pickle.load(file_pi)            
    full[0] = full[0][:, :, :, :, 0]
    print('retrieving ground truth coordinates')
    global norm_mean_skeleton
    norm_mean_skeleton = np.load(f'./prepared_data/{args.modality}_Skelet.npy')
    coord_gt = retrieves_gt_coord(ds)
    # intialize metrics
    global distance_l2
    global zdis
    global faux_pos
    global faux_neg
    global tot
    distance_l2 = []
    zdis = []
    faux_pos = []
    faux_neg = []
    tot = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.njoints)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'./weights/model_{args.modality}_att_{args.stacks}', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.njoints)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'./weights/model_{args.modality}_{args.stacks}', map_location='cpu')['model_weights'])


    ## Get the visualization resutls of the test set
    print(full[0].shape, full[1].shape)
    full_dataset_test = image_Dataset(image_paths=full[0],target_paths=full[1], use_flip = False)
    MRI_test_loader   = DataLoader(full_dataset_test, batch_size= 1, shuffle=False, num_workers=0)
    model.eval()
    for i, (input, target, vis) in enumerate(MRI_test_loader):
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        x      = full[0][i]
        prediction = extract_skeleton(input, output, target, Flag_save = False)
        
        
        #cv2.imwrite('./visualize/test1.png', np.sum(prediction[0], axis=0)*255)
        #cv2.imwrite('./visualize/test2.png', np.sum(f[0], axis=0)*255)

        prediction = np.sum(prediction[0], axis = 0)
        prediction = np.rot90(prediction,3)
        prediction = cv2.resize(prediction, (x.shape[0], x.shape[1]), interpolation=cv2.INTER_NEAREST)
        prediction_coordinates(prediction, coord_gt[i])
        #print(f'Image {i} is under process')

    print('distance med l2 and std ' + str(np.median(distance_l2)))
    print(np.std(distance_l2))
    print('distance med z and std ' + str(np.mean(zdis)))
    print(np.std(zdis))
    print('faux neg per image ', faux_neg)
    print('total number of points ' + str(np.sum(tot)))
    print('number of faux neg ' + str(np.sum(faux_neg)))
    print('number of faux pos ' + str(np.sum(faux_pos)))
    print('False negative percentage ' + str(np.sum(faux_neg)/ np.sum(tot)*100))
    print('False positive percentage ' + str(np.sum(faux_pos)/ np.sum(tot)*100))
    
##
def check_skeleton(cnd_sk, mean_skeleton):
    cnd_sk = np.array(cnd_sk)
    Normjoint = np.linalg.norm(cnd_sk[0]-cnd_sk[4])
    for idx in range(1, len(cnd_sk)):
        cnd_sk[idx] = (cnd_sk[idx] - cnd_sk[0]) / Normjoint
    cnd_sk[0] -= cnd_sk[0]
    
    return np.sum(np.linalg.norm(mean_skeleton[:len(cnd_sk)]-cnd_sk))
    

##    
idtest = 1
def extract_skeleton(inputs, outputs, target, Flag_save = False, target_th=0.5):
    global idtest
    outputs  = outputs.data.cpu().numpy()
    target  = target.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()
    skeleton_images = []
    for idx in range(outputs.shape[0]):    
        count_list = []
        Nch = 0
        center_list = {}
        while np.sum(np.sum(target[idx, Nch]))>0:
              Nch += 1       
        Final  = np.zeros((outputs.shape[0], Nch, outputs.shape[2], outputs.shape[3]))      
        for idy in range(Nch): 
            ych = outputs[idx, idy]
            ych = np.rot90(ych)
            ych = ych/np.max(np.max(ych))
            ych[np.where(ych<target_th)] = 0
            Final[idx, idy] = ych
            ych = np.where(ych>0, 1.0, 0)
            ych = np.uint8(ych)
            num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(ych)
            count_list.append(num_labels-1)
            center_list[str(idy)] = [t[::-1] for t in centers[1:]]
            
        ups = []
        for c in count_list:
            ups.append(range(c))
        combs = cartesian(ups)
        best_loss = np.Inf
        best_skeleton = []
        for comb in combs:
            cnd_skeleton = []
            for joint_idx, cnd_joint_idx in enumerate(comb):
                cnd_center = center_list[str(joint_idx)][cnd_joint_idx]
                cnd_skeleton.append(cnd_center)
            loss = check_skeleton(cnd_skeleton, norm_mean_skeleton)
            if best_loss > loss:
                best_loss = loss
                best_skeleton = cnd_skeleton
        Final2  = np.uint8(np.where(Final>0, 1, 0))
        cordimg = np.zeros(Final2.shape)
        hits = np.zeros_like(outputs[0])
        for i, jp, in enumerate(best_skeleton):
            jp = [int(t) for t in jp]
            hits[i, jp[0]-1:jp[0]+2, jp[1]-1:jp[1]+2] = [255, 255, 255]
            hits[i, :, :] = cv2.GaussianBlur(hits[i, :, :],(5,5),cv2.BORDER_DEFAULT)
            hits[i, :, :] = hits[i, :, :]/hits[i, :, :].max()*255
            cordimg[idx, i, jp[0], jp[1]] = 1
        
        for id_ in range(Final2.shape[1]):
            num_labels, labels_im = cv2.connectedComponents(Final2[idx, id_])
            for id_r in range(1, num_labels):
                if np.sum(np.sum((labels_im==id_r) * cordimg[idx, id_]) )>0:
                   labels_im = labels_im == id_r
                   continue
            Final2[idx, id_] = labels_im
        Final = Final * Final2           
                
        
        skeleton_images.append(hits)
        
    skeleton_images = np.array(skeleton_images)
    inputs = np.rot90(inputs, axes=(-2, -1))
    target = np.rot90(target, axes=(-2, -1))
    if Flag_save:
      save_test_results(inputs, skeleton_images, targets=target, name=idtest, target_th=0.5)
    idtest+=1
    return Final
    
##     
def save_test_results(inputs, outputs, targets, name='', target_th=0.5):
    clr_vis_Y = []
    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0,0], dtype=np.uint8)

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

    txt = f'./visualize/{name}_test_result.png'
    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(txt, res)



 
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verterbal disc labeling using pose estimation')

    ## Parameters
    parser.add_argument('--datapath', default='./prepared_data/prepared_testset', type=str,
                        help='Dataset address')                               
    parser.add_argument('--modality', default='t1', type=str, metavar='N',
                        help='Data modality')                                                

    parser.add_argument('--njoints', default=11, type=int,
                        help='Number of joints')
    parser.add_argument('--resume', default= False, type=bool,
                        help=' Resume the training from the last checkpoint') 
    parser.add_argument('--att', default= True, type=bool,
                        help=' Use attention mechanism') 
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')

    main(parser.parse_args())               