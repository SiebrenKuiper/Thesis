#!/bin/bash
# Windows subsystem for Linux (WSL), ubuntu 22.05, fMRIPrep 25.2.3.
# Limited by memory (24 GB), so processing one subject at a time.
for i in $(seq -w 2 77); do
    fmriprep-docker /mnt/d/MS_Thesis/Output/Image/BIDS /mnt/d/MS_Thesis/Output/Image/preproc/sub-${i} \
        participant \
        --participant-label sub-${i} \
        --fs-license-file mnt/d/MS_Thesis/Analysis/license.txt \
        --fs-no-reconall \
        --output-spaces MNI152NLin2009cAsym \
        --skull-strip-template OASIS30ANTs \
        --stop-on-first-crash
done
