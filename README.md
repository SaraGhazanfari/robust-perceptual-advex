# Robust Perceptual Aversarial Training
In this project we are using the [R-LPIPS](https://github.com/SaraGhazanfari/R-LPIPS) similarity metric to train a perceptual robust classifier. 
Our code is derived from [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex), 
which contains code and data for the ICLR 2021 paper "Perceptual Adversarial Robustness: Defense Against Unseen Threat Models".
All the functionalities in the [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex) are available in 
this project and our contribution is:
- Adding R-LPIPS similarity metric to distances.
- Using R-LPIPS for generating perceptual attacks R-PPGA and R-LPA.
- Using R-LPIPS for adversarially training the PAT model called R-PAT.

### Robust Perceptual Adversarial Training (R-PAT)

The script `adv_train.py` can be used to perform Robust Perceptual Adversarial Training (PAT) or to perform regular adversarial training. To train a ResNet-50 with r-lpips similarity metric on CIFAR-10:

    python adv_train.py --batch_size 50 --arch resnet50 --lpips_model r-lpips --dataset cifar --r_lpips_model_path path/to/r-lpips/model --attack "FastLagrangePerceptualAttack(model, bound=0.5, num_iterations=10, lpips_model='r-lpips', path='path/to/r-lpips/model')" --only_attack_correct

This will create a directory `data/logs`, which will contain [TensorBoard](https://www.tensorflow.org/tensorboard) logs and checkpoints for each epoch.

To train a ResNet-50 R-PAT on ImageNet-100:

    python adv_train.py --parallel 4 --batch_size 128 --dataset imagenet100 --dataset_path /path/to/ILSVRC2012 --arch resnet50 --lpips_model r-lpips r_lpips_model_path path/to/r-lpips/model --attack "FastLagrangePerceptualAttack(model, bound=0.25, num_iterations=10, lpips_model='r-lpips', path='path/to/r-lpips/model')" --only_attack_correct

### Generating Perceptual Adversarial Attacks

The script `generate_examples.py` will generate adversarially attacked images. For instance, to generate adversarial examples with the Perceptual Projected Gradient Descent (PPGD) and Lagrange Perceptual Attack (LPA) attacks on ImageNet, run:

    python generate_examples.py --dataset imagenet --arch resnet50 --checkpoint pretrained --batch_size 20 --shuffle --layout horizontal_alternate --dataset_path /path/to/ILSVRC2012 --output examples.png \
    "PerceptualPGDAttack(model, bound=0.5, num_iterations=40, lpips_model='r-lpips', path='path/to/r-lpips/model')" \
    "LagrangePerceptualAttack(model, bound=0.5, num_iterations=40, lpips_model='r-lpips', path='path/to/r-lpips/model')"
    
This will create an image called `examples.png` with three columns. The first is the unmodified original images from the ImageNet test set. The second and third contain adversarial attacks and magnified difference from the originals for the PPGD and LPA attacks, respectively.

#### Arguments

 - `--dataset` can be `cifar` for CIFAR-10, `imagenet100` for ImageNet-100, or `imagenet` for full ImageNet.
 - `--arch` can be `resnet50` (or `resnet34`, etc.) or `alexnet`.
 - `--checkpoint` can be `pretrained` to use the pretrained `torchvision` model. Otherwise, it should be a path to a pretrained model, such as those from the [robustness](https://github.com/MadryLab/robustness) library.
 - `--batch_size` indicates how many images to attack.
 - `--layout` controls the layout of the resulting image. It can be `vertical`, `vertical_alternate`, or `horizontal_alternate`.
 - `--output` specifies where the resulting image should be stored.
 - The remainder of the arguments specify attacks using Python expressions. See  the `perceptual_advex.attacks` and `perceptual_advex.perceptual_attacks` modules for a full list of available attacks and arguments for those attacks.
 
### Evaluation

The script `evaluate_trained_model.py` evaluates a model against a set of attacks. The arguments are similar to `generate_examples.py` (see above). For instance, to evaluate the torchvision pretrained ResNet-50 against PPGD and LPA, run:

    python evaluate_trained_model.py --dataset imagenet --arch resnet50 --checkpoint pretrained --batch_size 50 --dataset_path /path/to/ILSVRC2012 --output evaluation.csv \
    "PerceptualPGDAttack(model, bound=0.5, num_iterations=40, lpips_model='r-lpips', path='path/to/r-lpips/model')" \
    "LagrangePerceptualAttack(model, bound=0.5, num_iterations=40, lpips_model='r-lpips', path='path/to/r-lpips/model')"

#### CIFAR-10

The following command was used to evaluate CIFAR-10 classifiers for Tables 2, 6, 7, 8, and 9 in the paper:

    python evaluate_trained_model.py --dataset cifar --checkpoint /path/to/checkpoint.pt --arch resnet50 --batch_size 100 --output evaluation.csv \
    "NoAttack()" \
    "AutoLinfAttack(model, 'cifar', bound=8/255)" \
    "AutoL2Attack(model, 'cifar', bound=1)" \
    "StAdvAttack(model, num_iterations=100)" \
    "ReColorAdvAttack(model, num_iterations=100)" \
    "PerceptualPGDAttack(model, num_iterations=40, bound=0.5, lpips_model='alexnet_cifar', projection='newtons')" \
    "LagrangePerceptualAttack(model, num_iterations=40, bound=0.5, lpips_model='alexnet_cifar', projection='newtons')"\
    "PerceptualPGDAttack(model, num_iterations=40, bound=0.5, lpips_model='r-lpips', path='path/to/r-lpips/model', projection='newtons')" \
    "LagrangePerceptualAttack(model, num_iterations=40, bound=0.5, lpips_model='r-lpips', path='path/to/r-lpips/model', projection='newtons')"

#### ImageNet-100

The following command was used to evaluate ImageNet-100 classifiers for Table 3 in the paper, which shows the robustness of various models against several attacks at the medium perceptibility bound:

    python evaluate_trained_model.py --dataset imagenet100 --dataset_path /path/to/ILSVRC2012 --checkpoint /path/to/checkpoint.pt --arch resnet50 --batch_size 50 --output evaluation.csv \
    "NoAttack()" \
    "AutoLinfAttack(model, 'imagenet100', bound=4/255)" \
    "AutoL2Attack(model, 'imagenet100', bound=1200/255)" \
    "JPEGLinfAttack(model, 'imagenet100', bound=0.125, num_iterations=200)" \
    "StAdvAttack(model, bound=0.05, num_iterations=200)" \
    "ReColorAdvAttack(model, bound=0.06, num_iterations=200)" \
    "PerceptualPGDAttack(model, bound=0.5, lpips_model='alexnet', num_iterations=40)" \
    "LagrangePerceptualAttack(model, bound=0.5, lpips_model='alexnet', num_iterations=40)"\
    "PerceptualPGDAttack(model, bound=0.5, lpips_model='alexnet', num_iterations=40, lpips_model='r-lpips', path='path/to/r-lpips/model')" \
    "LagrangePerceptualAttack(model, bound=0.5, lpips_model='alexnet', num_iterations=40, lpips_model='r-lpips', path='path/to/r-lpips/model')"

## Contact

For questions about the code, please contact sg7457@nyu.edu.
