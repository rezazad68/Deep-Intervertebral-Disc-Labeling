#===================================================
## Author: Reza Azad (rezazad68@gmail.com)
#===================================================

from shutil import copyfile
import os
import numpy as np

ADD =  "/reza/data/data-multi-subject/derivatives/labels/"
ADD2 = "/reza/data/data-multi-subject/"

Destination_path = 'dataset/'

list_dir = os.listdir(ADD)
Total = 0
for idx in range (len(list_dir)):   
    ## Copy the T1 disc label   
    src1 = ADD+list_dir[idx] + '/anat/'+list_dir[idx]+'_T1w_labels-disc-manual.nii.gz'
    dst1 = list_dir[idx] + '/'         +list_dir[idx]+'_T1w_labels-disc-manual.nii.gz'
    
    ## Copy the T2 disc label   
    src2 = ADD+list_dir[idx] + '/anat/'+list_dir[idx]+'_T2w_labels-disc-manual.nii.gz'
    dst2 = list_dir[idx] + '/'         +list_dir[idx]+'_T2w_labels-disc-manual.nii.gz'    
    
    ## Copy the T1 file   
    src3 = ADD2+list_dir[idx] + '/anat/'+list_dir[idx]+'_T1w.nii.gz'
    dst3 = list_dir[idx] + '/'          +list_dir[idx]+'_T1w.nii.gz'    
    
    ## Copy the T2 file   
    src4 = ADD2+list_dir[idx] + '/anat/'+list_dir[idx]+'_T2w.nii.gz'
    dst4 = list_dir[idx] + '/'          +list_dir[idx]+'_T2w.nii.gz'            
    
    if os.path.exists(src1):
       if not os.path.exists('dataset/'+list_dir[idx]):
          os.makedirs(Destination_path+list_dir[idx])
       copyfile(src1, Destination_path+dst1)
       copyfile(src2, Destination_path+dst2)
       copyfile(src3, Destination_path+dst3)
       copyfile(src4, Destination_path+dst4)
       Total += 1

print(f'Total number of {Total} subject selected')
