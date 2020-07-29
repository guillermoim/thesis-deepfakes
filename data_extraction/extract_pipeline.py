import argparse

from extract_landmarks import extract_landmarks
from extract_boxes import extract_original_boxes
from extract_faces import extract_faces_from_dir
from extract_ssim_masks import extract_ssim_masks


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder where videos are store.')
    parser.add_argument('--out', type=str, required=True, help='Path to the folder where results will be stored.')
    args = parser.parse_args()
    src_dir = args.data_path
    dst_dir = args.out
    extract_original_boxes(src_dir, dst_dir)
    extract_faces_from_dir(src_dir, dst_dir)
    extract_landmarks(dst_dir)
    extract_ssim_masks(dst_dir)


if __name__ == '__main__':
    main()