source ../torch/bin/activate

PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py --ngpu 6  --distributed --n_workers 6 > outputs/effnetb3_tum_101_v0_fullDA.txt
PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py --ngpu 6  --data_augment no_da --distributed --n_workers 6 > outputs/effnetb3_tum_101_v0_noDA.txt
PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py --ngpu 6  --data_augment simple_da --distributed --n_workers 6 > outputs/effnetb3_tum_101_v0_simpleDA.txt
PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py --ngpu 6  --data_augment occlusions_da --distributed --n_workers 6 > outputs/effnetb3_tum_101_v0_occDA.txt

#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py -s 101 -m xception -batch_size 8 -epoch_size 100 -epochs 30 --ngpu 6 --n_workers 6 -v 0 --amp True --data_path ../datasets/mtcnn > outputs/xception_tum_101_v0.txt
#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py -s 101 -m efficientnet-b3 -batch_size 8 -epoch_size 100 -epochs 30 --ngpu 6 --n_workers 6 -v 0 --amp True --data_path ../datasets/mtcnn > outputs/effnetb3_tum_101_v0.txt
#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py -s 101 -m efficientnet-b5 -batch_size 8 -epoch_size 100 -epochs 30 --ngpu 6 --n_workers 6 -v 0 --amp True --data_path ../datasets/mtcnn > outputs/effnetb5_tum_101_v0.txt
#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py -s 101 -m regnety-1.6GF -batch_size 8 -epoch_size 100 -epochs 30 --ngpu 6 --n_workers 6 -v 0 --amp True --data_path ../datasets/mtcnn > outputs/regnety-1.6GF_tum_101_v0.txt
#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train_ff.py -s 1 -m xception -batch_size 8 -epoch_size 100 -epochs 30 --ngpu 6 --n_workers 6 -v 0 --amp True --data_path ../datasets/mtcnn > outputs/test_sigmoids.txt
