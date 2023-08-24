"# OptimalQT" 

## Split data into High Quality and Low Quality:
python split_imagenet_by_quality.py \<imagenet path\>
    
## code to launch training for imagenet as per torchvision's recipe
torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model resnet50-lps --batch-size 128 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints" --resume