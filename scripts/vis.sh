CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
python demo/demo.py --config-file configs/DPText_DETR/MPSC/R_50_poly_debug.yaml \
 --input ./datasets/MPSC/image/train/MPSC_img_1175.jpg --output ./vis/mpsc_1175/ \
 --opts MODEL.WEIGHTS ./output/r_50_poly/MPSC/annotation_consistency/baseline_revised/model_0001999.pth

# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
# python demo/demo.py --config-file configs/DPText_DETR/MPSC/R_50_poly_debug.yaml \
#  --input ./datasets/MPSC/image/test/ --output ./vis/reann_rec_emb_v38/ \
#  --opts MODEL.WEIGHTS ./output/r_50_poly/MPSC/reann_rec_emb_v38/model_0025999.pth \

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 \
# python demo/demo.py --config-file configs/DPText_DETR/MPSC/R_50_poly.yaml \
#  --input ./datasets/MPSC/image/test_contrast_10/ --output ./vis/mpsc_low_contrast_10/ \
#  --opts MODEL.WEIGHTS ./output/r_50_poly/MPSC/finetune/model_final.pth

#  python demo/demo.py --config-file configs/DPText_DETR/TotalText/R_50_poly.yaml \
#  --input ./datasets/totaltext/test_images_rotate/ --output ./vis/totaltext/ \
#  --opts MODEL.WEIGHTS ./adet/checkpoint/totaltext_final.pth
