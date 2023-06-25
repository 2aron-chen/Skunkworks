#!/bin/bash

fname_dat="T1_3.dat"
fname_nii="T1_3.nii"

output_name="`basename ${fname_dat} | cut -d '.' -f 1`.nii"
echo "output_name is: ${output_name}"

header_size="352"

head --bytes=${header_size} ${fname_nii} > ${output_name}
cat ${fname_dat} >> ${output_name}



exit 0



