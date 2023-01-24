python3 metrics/compute_fid.py ${SRC} ${REF}
python3 metrics/compute_miou.py --input ${SRC} --meta-file ${META}
python3 metrics/compute_clip.py --input ${SRC} --meta-file ${META}