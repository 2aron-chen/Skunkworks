#!/bin/bash

exe="/home/kecskemeti/local/bin/fit_signal"

fitting_options=" -fit_signal -t1fit_mpnrage_vfa -fit_frequency 20"
fitting_options+=" -fitting_num_slices 8"
fitting_options+=" -threshold_mask 0.002 -central_blob_mask -fsize_mask 2"
fitting_options+=" -fitting_num_passes 4 -fsize_b1 7 -fsize_inv_eff 9 -nlm_sigma 900 -nlm_width 9"
fitting_options+=" -robust_init"

echo "${exe} -t1fit_mpnrage_vfa -pca ${fitting_options}"

${exe} -t1fit_mpnrage_vfa -pca ${fitting_options}
