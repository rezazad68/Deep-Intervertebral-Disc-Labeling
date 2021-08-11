
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}Started${reset}"

for d in ./*/; do
    sn=${d:2:-1};
    
    if [[ "$sn" != "sub-"* ]]; then
        echo "${red} --> Ignored ${sn}...${reset}"
        continue
    fi

    echo "${red}Processing ${sn} directory...${reset}";
    
    cd $sn

    sct_deepseg_sc -i "${sn}_T2w.nii.gz" -c t2
    # the output is -> <folder_name>_T2w_seg.nii.gz

    sct_straighten_spinalcord -i "${sn}_T2w.nii.gz" -s "${sn}_T2w_seg.nii.gz"
    # the outputs are -> [
    #     warp_curve2straight.nii.gz,
    #     warp_straight2curve.nii.gz,
    #     <folder_name>_T2w_straight.nii.gz,
    #     straight_ref.nii.gz
    # ]

    sct_apply_transfo -i "${sn}_T2w_labels-disc-manual.nii.gz" -d straight_ref.nii.gz -w warp_curve2straight.nii.gz -x label
    # the output is -> <folder_name>_T2w_labels-disc-manual_reg.nii.gz

    cd ..;
    echo "${green}Processed ${sn} directory.${reset}";
done




