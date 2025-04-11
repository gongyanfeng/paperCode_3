#python train.py --dataset my --isize 32 --niter 15 --ndf 32 --ngf 32
python test.py --dataset tofd --phase test  --isize 128 --ndf 128 --ngf 128 --save_test_images --load_weights
