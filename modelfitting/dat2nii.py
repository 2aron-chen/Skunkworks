from subprocess import call
import os
import time
for file in sorted(os.listdir('/scratch/mrphys/denoised/')):
    if "." in file:
        continue
    t0 = time.time()
    print(f"Start processing {file}")
    status = call("nice bash /study/mrphys/skunkworks/kk/modelfitting/dat2nii.sh",cwd=f"/scratch/mrphys/denoised/{file}",shell=True)
    print(f"Completed file {file} with exit status {status}")
    eta = round((time.time()-t0)/60, 2)
    print(f"ETA = {eta}")