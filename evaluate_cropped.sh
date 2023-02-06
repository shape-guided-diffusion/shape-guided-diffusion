CROPPED_SRC=$(python3 metrics/crop_images.py --input ${SRC} --meta-file ${META})
CROPPED_REF=$(python3 metrics/crop_images.py --input ${REF} --meta-file ${META})
python3 metrics/compute_fid.py ${CROPPED_SRC} ${CROPPED_REF}
python3 metrics/compute_miou.py --input ${CROPPED_SRC} --meta-file ${META}
python3 metrics/compute_clip.py --input ${CROPPED_SRC} --meta-file ${META}
rm -rf $CROPPED_SRC
rm -rf $CROPPED_REF