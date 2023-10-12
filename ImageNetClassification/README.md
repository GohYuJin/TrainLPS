# TrainLPS 

Code in the learn_poly_sampling module was extracted from https://github.com/raymondyeh07/learnable_polyphase_sampling
Training script was adapted from torchvision training recipes https://github.com/pytorch/vision/tree/main/references/classification
    
## code to launch training for imagenet as per torchvision's recipe
torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model resnet50-lps --batch-size 128 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints" --save-every=5 --workers=8 --resume checkpoints/checkpoint.pth

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model resnetlite50type3-lps --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpointstype3" --save-every=5 --workers=8 --resume checkpointstype3/checkpoint.pth

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model resnet50halved --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet50halved" --save-every=5 --workers=4 --resume "checkpoints-resnet50halved/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model efficientnet_b4 --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints-efficientnetb4" --save-every=5 --workers=4 --resume "checkpoints-efficientnetb4/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model resnet101halved --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet101halved" --save-every=5 --workers=4 --resume "checkpoints-resnet101halved/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --model resnet50-aps --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet50aps" --save-every=5 --workers=4 --resume "checkpoints-resnet50aps/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --train-csv=halvedImageNet.csv --model resnet50-lps --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet50lps-reduceddata" --save-every=5 --workers=4 --resume "checkpoints-resnet50lps-reduceddata/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --train-csv=halvedImageNet.csv --model resnet50 --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet50-reduceddata" --save-every=5 --workers=4 --resume "checkpoints-resnet50-reduceddata/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --train-csv=tenthedImageNet.csv --model resnet50 --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet50-10xreduceddata" --save-every=5 --workers=4 --resume "checkpoints-resnet50-10xreduceddata/checkpoint.pth"

torchrun --nproc_per_node=4 train-resnet-lps.py --data-path $IMAGENET_PATH --train-csv=tenthedImageNet.csv --model resnet50-lps --batch-size 256 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 200 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 \
--ra-sampler --ra-reps=4 --output-dir="checkpoints-resnet50lps-10xreduceddata" --save-every=5 --workers=4 --resume "checkpoints-resnet50lps-10xreduceddata/checkpoint.pth"


