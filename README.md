# L2M-GAN
> Unofficial PyTorch implementation of L2M-GAN.

## Steps
1. Download the [wing.ckpt](https://www.dropbox.com/s/tjxpypwpt38926e/wing.ckpt) and put it in `./archive/models/`.
2. Download the CelebA-HQ dataset from [here](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv).
3. Use the script `./bin/split_celeba.py` to generate the dataset split and then put it in `./archive/`.
4. Make the shell script executable: `chmod u+x ./scripts/train.py`
5. Execute the shell script: `./scripts/train.py`

## TODOs
+ [x] Implement the models.
+ [x] Implement the loss functions.
+ [x] Make it runnable.
+ [x] Start the experiment.

## Results
Wait for the experiment to complete.