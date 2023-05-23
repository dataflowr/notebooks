# [Dataflowr: Deep Learning DIY](https://www.dataflowr.com/)

[![Dataflowr](https://raw.githubusercontent.com/dataflowr/website/master/_assets/dataflowr_logo.png)](https://dataflowr.github.io/website/)

Code and notebooks for the deep learning course [dataflowr](https://www.dataflowr.com/). Here is the schedule followed at école polytechnique in 2023:

## :sunflower:Session:one: Finetuning VGG

>- [Module 1 - Introduction & General Overview](https://dataflowr.github.io/website/modules/1-intro-general-overview/)
Slides + notebook Dogs and Cats with VGG + Practicals (more dogs and cats) 
<details>
  <summary>Things to remember</summary>

> - you do not need to understand everything to run a deep learning model! But the main goal of this course will be to come back to each step done today and understand them...
> - to use the dataloader from Pytorch, you need to follow the API (i.e. for classification store your dataset in folders)
> - using a pretrained model and modifying it to adapt it to a similar task is easy. 
> - if you do not understand why we take this loss, that's fine, we'll cover that in Module 3.
> - even with a GPU, avoid unnecessary computations!

</details>

## :sunflower:Session:two: PyTorch tensors and Autodiff

>- [Module 2a - PyTorch tensors](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)
>- [Module 2b - Automatic differentiation](https://dataflowr.github.io/website/modules/2b-automatic-differentiation/) + Practicals
>- MLP from scratch start of [HW1](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/) 
>- [another look at autodiff with dual numbers and Julia](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb)
<details>
  <summary>Things to remember</summary>

>- Pytorch tensors = Numpy on GPU + gradients!
>- in deep learning, [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is used everywhere. The rules are the same as for Numpy.
>- Automatic differentiation is not only the chain rule! Backpropagation algorithm (or dual numbers) is a clever algorithm to implement automatic differentiation...

 </details>

## :sunflower:Session:three: 
> - [Module 3 - Loss function for classification](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/) 
> - [Module 4 - Optimization for deep learning](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)
> - [Module 5 - Stacking layers](https://dataflowr.github.io/website/modules/5-stacking-layers/) and overfitting a MLP on CIFAR10: [Stacking_layers_MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/Stacking_layers_MLP_CIFAR10.ipynb)
> - [Module 6: Convolutional neural network](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/)
> - how to regularize with dropout and uncertainty estimation with MC Dropout: [Module 15 - Dropout](https://dataflowr.github.io/website/modules/15-dropout/)
<details>
  <summary>Things to remember</summary>

>- Loss vs Accuracy. Know your loss for a classification task!
>- know your optimizer (Module 4)
>- know how to build a neural net with torch.nn.module (Module 5)
>- know how to use convolution and pooling layers (kernel, stride, padding)
>- know how to use dropout 

</details>

## :sunflower:Session:four:
> - [Module 7 - Dataloading](https://dataflowr.github.io/website/modules/7-dataloading/)
> - [Module 8a - Embedding layers](https://dataflowr.github.io/website/modules/8a-embedding-layers/)
> - [Module 8b - Collaborative filtering](https://dataflowr.github.io/website/modules/8b-collaborative-filtering/) and build your own recommender system: [08_collaborative_filtering_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_empty.ipynb) (on a larger dataset [08_collaborative_filtering_1M.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_1M.ipynb))
> - [Module 8c - Word2vec](https://dataflowr.github.io/website/modules/8c-word2vec/) and build your own word embedding [08_Word2vec_pytorch_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Word2vec_pytorch_empty.ipynb)
> - [Module 16 - Batchnorm](https://dataflowr.github.io/website/modules/16-batchnorm/) and check your understanding with [16_simple_batchnorm_eval.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_simple_batchnorm_eval.ipynb) and more [16_batchnorm_simple.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_batchnorm_simple.ipynb)
> - [Module 17 - Resnets](https://dataflowr.github.io/website/modules/17-resnets/)

<details>
  <summary>Things to remember</summary>

> - know how to use dataloader
> - to deal with categorical variables in deep learning, use embeddings
> - in the case of word embedding, starting in an unsupervised setting, we built a supervised task (i.e. predicting central / context words in a window) and learned the representation thanks to negative sampling
> - know your batchnorm
> - architectures with skip connections allows deeper models

</details>

## :sunflower:Session:five:
> - [Module 9a: Autoencoders](https://dataflowr.github.io/website/modules/9a-autoencoders/) and code your noisy autoencoder [09_AE_NoisyAE.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/09_AE_NoisyAE.ipynb)
> - [Module 10: Generative Adversarial Networks]() and code your GAN, Conditional GAN and InfoGAN [10_GAN_double_moon.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module10/10_GAN_double_moon.ipynb)
> - [Module 13: Siamese Networks and Representation Learning](https://dataflowr.github.io/website/modules/13-siamese/)

## :sunflower:Session:six:
> - [Module 11a - Recurrent Neural Networks theory](https://dataflowr.github.io/website/modules/11a-recurrent-neural-networks-theory/)
> - [Module 11b - Recurrent Neural Networks practice](https://dataflowr.github.io/website/modules/11b-recurrent-neural-networks-practice/) and predict engine failure with [11\_predicitions\_RNN\_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_predicitions_RNN_empty.ipynb)
> - [Module 11c - Batches with sequences in Pytorch](https://dataflowr.github.io/website/modules/11c-batches-with-sequences/)

## :sunflower:Session:seven:
TBC

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/marc_lelarge.svg?style=social&label=Follow%20%40marc_lelarge)](https://twitter.com/marc_lelarge) 
# :sunflower: All notebooks

- [**Module 1: Introduction & General Overview**](https://dataflowr.github.io/website/modules/1-intro-general-overview/) 
    - Intro: finetuning VGG for dogs vs cats [01_intro.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/01_intro.ipynb)
    - Practical: Using CNN for more dogs and cats [01_practical_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/01_practical_empty.ipynb) and its solution [01_practical_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/sol/01_practical_sol.ipynb)
- [**Module 2: Pytorch tensors and automatic differentiation**](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)
    - Basics on PyTorch tensors and automatic differentiation [02a_basics.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb)
    - Linear regression from numpy to pytorch [02b_linear_reg.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb)
    - Practical: implementing backprop from scratch [02_backprop.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02_backprop.ipynb) and its solution [02_backprop_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/sol/02_backprop_sol.ipynb)
    - Bonus: intro to JAX: autodiff the functional way [autodiff_functional_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_empty.ipynb) and its solution [autodiff_functional_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_sol.ipynb)
    - Bonus: Linear regression in JAX [linear_regression_jax.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb)
    - Bonus: automatic differentiation with dual numbers [AD_with_dual_numbers_Julia.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb)
- [**Homework 1: MLP from scratch**](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/)
    - [hw1_mlp.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW1/hw1_mlp.ipynb) and its solution [hw1_mlp_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW1/sol/hw1_mlp_sol.ipynb)
- [**Module 3: Loss functions for classification**](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/)
    - An explanation of underfitting and overfitting with polynomial regression [03_polynomial_regression.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb)
- [**Module 4: Optimization for deep leaning**](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)
    - Practical: code Adagrad, RMSProp, Adam, AMSGrad [04_gradient_descent_optimization_algorithms_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module4/04_gradient_descent_optimization_algorithms_empty.ipynb) and its solution [04_gradient_descent_optimization_algorithms_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module4/sol/04_gradient_descent_optimization_algorithms_sol.ipynb)
- [**Module 5: Stacking layers**](https://dataflowr.github.io/website/modules/5-stacking-layers/)
    - Practical: overfitting a MLP on CIFAR10 [Stacking_layers_MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/Stacking_layers_MLP_CIFAR10.ipynb) and its solution [MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/sol/MLP_CIFAR10.ipynb)
- [**Module 6: Convolutional neural network**](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/)
    - Practical: build a simple digit recognizer with CNN [06_convolution_digit_recognizer.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb)
- [**Homework 2: Class Activation Map and adversarial examples**](https://dataflowr.github.io/website/homework/2-CAM-adversarial/)
    - [HW2_CAM_Adversarial.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW2/HW2_CAM_Adversarial.ipynb)

- [**Module 8: Embedding layers**](https://dataflowr.github.io/website/modules/8a-embedding-layers/), [**Collaborative filtering**](https://dataflowr.github.io/website/modules/8b-collaborative-filtering/) and [**Word2vec**](https://dataflowr.github.io/website/modules/8c-word2vec/)
    - Practical: Collaborative filtering with Movielens 100k dataset [08_collaborative_filtering_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_empty.ipynb)
    - Practical: Refactoring code, collaborative filtering with Movielens 1M dataset [08_collaborative_filtering_1M.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_collaborative_filtering_1M.ipynb)
    - Practical: Word Embedding (word2vec) in PyTorch [08_Word2vec_pytorch_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Word2vec_pytorch_empty.ipynb)
    - Finding Synonyms and Analogies with Glove [08_Playing_with_word_embedding.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module8/08_Playing_with_word_embedding.ipynb)
- [**Module 9a: Autoencoders**](https://dataflowr.github.io/website/modules/9-autoencoders/)
    - Practical: denoising autoencoder (with convolutions and transposed convolutions) [09_AE_NoisyAE.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/09_AE_NoisyAE.ipynb)
- [**Module 9b - UNets**](https://dataflowr.github.io/website/modules/9b-unet/)
  - UNet for image segmentation [UNet_image_seg.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/UNet_image_seg.ipynb)
- [**Module 9c - Flows**](https://dataflowr.github.io/website/modules/9c-flows/) 
  - implementing Real NVP [Normalizing_flows_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/Normalizing_flows_empty.ipynb) and its solution [Normalizing_flows_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module9/Normalizing_flows_sol.ipynb)
- [**Module 10 - Generative Adversarial Networks**](https://dataflowr.github.io/website/modules/10-generative-adversarial-networks/)
  - Conditional GAN and InfoGAN [10_GAN_double_moon.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module10/10_GAN_double_moon.ipynb)
- [**Module 11 - Recurrent Neural Networks**](https://dataflowr.github.io/website/modules/11b-recurrent-neural-networks-practice/) and [**Batches with sequences in Pytorch**](https://dataflowr.github.io/website/modules/11c-batches-with-sequences/)
  - notebook used in the theory course: [11_RNN.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_RNN.ipynb)
  - predicting engine failure with RNN [11_predicitions_RNN_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module11/11_predicitions_RNN_empty.ipynb)
- [**Module 12 - Attention and Transformers**](https://dataflowr.github.io/website/modules/12-attention/)
  - Correcting the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) on attention in seq2seq: [12_seq2seq_attention.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/12_seq2seq_attention.ipynb) and its [solution](https://github.com/dataflowr/notebooks/blob/master/Module12/12_seq2seq_attention_solution.ipynb)
  - building a simple transformer block and thinking like transformers: [GPT_hist.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist.ipynb) and its [solution](https://github.com/dataflowr/notebooks/blob/master/Module12/GPT_hist_sol.ipynb)
- [**Module 13 - Siamese Networks and Representation Learning**](https://dataflowr.github.io/website/modules/13-siamese/)
  - learning embeddings with contrastive loss: [13_siamese_triplet_mnist_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module13/13_siamese_triplet_mnist_empty.ipynb) 
- [**Module 15 - Dropout**](https://dataflowr.github.io/website/modules/15-dropout/)
  - Dropout on a toy dataset: [15a_dropout_intro.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module15/15a_dropout_intro.ipynb)
  - playing with dropout on MNIST: [15b_dropout_mnist.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module15/15b_dropout_mnist.ipynb)
- [**Module 16 - Batchnorm**](https://dataflowr.github.io/website/modules/16-batchnorm/)
  - impact of batchnorm: [16_batchnorm_simple.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_batchnorm_simple.ipynb)
  - Playing with batchnorm without any training: [16_simple_batchnorm_eval.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module16/16_simple_batchnorm_eval.ipynb)
- [**Module 18a - Denoising Diffusion Probabilistic Models**](https://dataflowr.github.io/website/modules/18a-diffusion/)
  - Denoising Diffusion Probabilistic Models for MNIST: [ddpm_nano_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_empty.ipynb) and its solution [ddpm_nano_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_nano_sol.ipynb)
  - Denoising Diffusion Probabilistic Models for CIFAR10: [ddpm_micro_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb)
- [**Module - Deep Learning on graphs**](https://dataflowr.github.io/website/modules/graph0/)
  - Inductive bias in GCN: a spectral perspective [GCN_inductivebias_spectral.ipynb](https://github.com/dataflowr/notebooks/blob/master/graphs/GCN_inductivebias_spectral.ipynb) and for colab [GCN_inductivebias_spectral-colab.ipynb](https://github.com/dataflowr/notebooks/blob/master/graphs/GCN_inductivebias_spectral-colab.ipynb)
  - Graph ConvNets in PyTorch [spectral_gnn.ipynb](https://github.com/dataflowr/notebooks/blob/master/graphs/spectral_gnn.ipynb)
-  **NERF**
   -  PyTorch Tiny NERF [tiny_nerf_extended.ipynb](https://github.com/dataflowr/notebooks/blob/master/nerf/tiny_nerf_extended.ipynb)


## Usage

If you want to run locally, follow the instructions of [Module 0 - Running the notebooks locally](https://dataflowr.github.io/website/modules/0-sotfware-installation/)

## 2020 version of the course
Archives are available on the archive-2020 branch.