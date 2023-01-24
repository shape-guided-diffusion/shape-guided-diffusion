import sys
from cleanfid import fid

IMAGE_SIZE = 512

def main():    
    score = fid.compute_fid(sys.argv[1], sys.argv[2], dataset_res=IMAGE_SIZE)
    print("FID:", score)

if __name__ == "__main__":
    main()