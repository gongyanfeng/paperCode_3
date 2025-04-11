#python train.py --dataset my --isize 32 --niter 15 --ndf 32 --ngf 32 --batchsize 32
#python train.py --dataset tofd --isize 128 --niter 400 --ndf 128 --ngf 128 --save_test_images --display --resume "/home/ubuntu/f/gyf/weld_detection/ganomaly-master/output/ganomaly/tofd/train/weights"

#python train.py --dataset tofd --isize 128 --niter 400 --ndf 128 --ngf 128  --batchsize 32  --resume "/home/ubuntu/f/gyf/weld_detection/ganomaly-master/output/ganomaly/tofd/train/weights"

python train.py --dataset tofd --isize 128 --niter 200 --ndf 128 --ngf 128  --batchsize 32
