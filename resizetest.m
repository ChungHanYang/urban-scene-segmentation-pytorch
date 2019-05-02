fileName = ['/Volumes/Transcend/EEE598_image_understanding/leftImg8bit_trainvaltest/leftImg8bit/train/aachen_000000_000019_leftImg8bit.png'];
x = imread(fileName);
x = imresize(x,[512,1024],'bilinear', 'AntiAliasing', false);
