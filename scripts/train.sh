CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 \
python tools/train_net.py --config-file configs/ContraText-DETR/MPSC/R_50.yaml \
--num-gpus 4 \
--dist-url tcp://127.0.0.1:50235

