CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 \
python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly_640.yaml \
 --eval-only MODEL.WEIGHTS /home/ysjeong/workspace/OCR/DPText-DETR/output/r_50_poly/MPSC/contrastext-detr_640/model_0029999.pth


# python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly_debug.yaml \
#  --eval-only MODEL.WEIGHTS /home/ysjeong/workspace/OCR/DPText-DETR/output/r_50_poly/MPSC/reann_rec_emb_v38_loss_1/model_final.pth

# --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/reann_rec_emb_v38/model_0025999.pth


#  --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/reann_rec_v1/model_0026999.pth

# python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
#  --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/finetune/model_final.pth

# python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
#  --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/synthsize_tt_sample_v6/model_final.pth

# python tools/train_net.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
#  --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/finetune/model_final.pth
#  --eval-only MODEL.WEIGHTS ./output/r_50_poly/MPSC/DPText_DETR_aiaw_loss_10/model_final.pth

# python tools/train_net.py --config-file configs/DPText_DETR/TotalText/R_50_poly.yaml \
#  --eval-only MODEL.WEIGHTS ./adet/checkpoint/totaltext_final.pth