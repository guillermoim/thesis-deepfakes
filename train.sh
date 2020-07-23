source ../torch/bin/activate

#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train.py -s 111 -m xception -batch_size 32 -epoch_size 500 -epochs 30 -path_to_model models/xception_v0_111.pth --ngpu 6

#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train.py -s 111 -m effnet-b3 -batch_size 32 -epoch_size 2500 -epochs 30 -path_to_model models/effnetb3_v0_111.pth --ngpu 6 --n_workers 8

#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train.py -s 222 -m xception -batch_size 12 -epoch_size 2500 -epochs 15 -path_to_model models/xception_v0_111.pth --ngpu 4 --n_workers 6 -v 0 > outputs/a.txt

#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2" python -m torch.distributed.launch --nproc_per_node=2 train.py -s 111 -m effnet-b5 -batch_size 10 -epoch_size 8 -epochs 15 -path_to_model models/xception_v0_111.pth --ngpu 2 --n_workers 6 -v 0 > outputs/a.txt

PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train.py -s 333 -m effnet-b3 -batch_size 32 -epoch_size 1100 -epochs 30 -path_to_model models/effnetb3_v2_1024.pth --ngpu 6 --n_workers 6 -v 0 > outputs/b.txt

#PYTHONPATH=. CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" python -m torch.distributed.launch --nproc_per_node=6 train.py -s 1024 -m effnet-b5 -batch_size 12 -epoch_size 2500 -epochs 30 -path_to_model models/effnetb3_v0_1024.pth --ngpu 6 --n_workers 6 -v 0 --optim_step 2 > outputs/c.txt

