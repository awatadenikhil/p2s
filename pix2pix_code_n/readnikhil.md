Sr. No 	Steps
Dataset Preparation	Dataset Preparation (Paired)
	Pix2Pix requires paired images (input on the left, target on the right, concatenated into one image).
	Example (0001.jpg):
	
	-------------------------------
|    Photo    |    Sketch     |
-------------------------------
	datasets/
	 └── photo2sketch/
	      ├── train/
	      │    ├── 0001.jpg
	      │    ├── 0002.jpg
	      │    └── ...
	      └── test/
	           ├── 1001.jpg
	           ├── 1002.jpg
	           └── ...
	Create a dataset using command python convertdataset.py
Clone repo	git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
	cd pytorch-CycleGAN-and-pix2pix
	pip install -r requirements.txt
Place dataset folder	From command :python convertdataset.py get get folder photo2sketch . Put folder photo2sketch  into datasets
	
Train 	python train.py --dataroot ./datasets/photo2sketch_full --name photo2sketch_full --model pix2pix --direction AtoB --batch_size 1 --load_size 256 --crop_size 256


	• --dataroot = path to your dataset
	• --model pix2pix = use pix2pix model
	• --direction AtoB = left side is photo (A), right side is sketch (B). Swap if needed.
Test	python test.py --dataroot ./datasets/photo2sketch_full --name photo2sketch_full --model pix2pix --direction AtoB --num_test 50
	
Results will be saved in:
	
	./results/photo2sketch_pix2pix/test_latest/images/




	-----------------
	python train.py --dataroot ./datasets/p2s_data --name p2s_data --model pix2pix --direction AtoB --batch_size 1 --load_size 256 --crop_size 256
	python test.py --dataroot ./datasets/p2s_data --name p2s_data --model pix2pix --direction AtoB --num_test 50


	----
	python train.py --dataroot ./datasets/i2s_v3 --name i2s_v3 --model pix2pix --direction AtoB --batch_size 1 --load_size 256 --crop_size 256
	python test.py --dataroot ./datasets/i2s_v3 --name i2s_v3 --model pix2pix --direction AtoB --num_test 50

-----------------------single pic

python test.py --dataroot ./datasets/data/ --name i2s_v3 --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch

