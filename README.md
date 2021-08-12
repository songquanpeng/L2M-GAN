# L2M-GAN
> Unofficial PyTorch implementation of L2M-GAN.

## Steps
1. Download the [wing.ckpt](https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt) and put it in `./archive/models/`.
2. Download the CelebA-HQ dataset from [here](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv).
3. Use the script `./bin/split_celeba.py` to generate the dataset split, rename the generated folder to `celeba_hq_smiling` 
and then put it in `./archive/`.
4. Make the shell script executable: `chmod u+x ./scripts/train.py`
5. Execute the shell script: `./scripts/train.py`

## TODOs
+ [x] Implement the models.
+ [x] Implement the loss functions.
+ [x] Make it runnable.
+ [x] Start the experiments.

## Results
### Experiment #1: Attribute Smiling
Final best FID: 16.93 (100k iterations)
![default_setting_smiling_test_100000](https://user-images.githubusercontent.com/39998050/129124777-81ea0d92-0733-433f-a8f9-935b2c4d8930.jpg)

The first row is the origin images, the second row is the smiling one and the third row is the non-smiling one.

### Experiment #2: Attribute Gender
Final best FID: 33.21 (100k iterations)
![default_setting_gender_test_100000](https://user-images.githubusercontent.com/39998050/129124711-7aaf8e70-e119-465e-97b6-df46ea66c54c.jpg)

The first row is the origin images, the second row is the female one and the third row is the male one.
