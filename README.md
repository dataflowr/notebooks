# [Dataflowr: Deep Learning DIY](https://www.dataflowr.com/)

[![Dataflowr](https://raw.githubusercontent.com/dataflowr/website/master/_assets/dataflowr_logo.png)](https://dataflowr.github.io/website/)

Code and notebooks for the deep learning course [dataflowr](https://www.dataflowr.com/)

:sunflower:Session:one:
> [Module 1 - Introduction & General Overview](https://dataflowr.github.io/website/modules/1-intro-general-overview/)
Slides + notebook Dogs and Cats with VGG + Practicals (more dogs and cats) 

> Things to remember:
> - you do not need to understand everything to run a deep learning model! But the main goal of this course will be to come back to each step done today and understand them...
> - to use the dataloader from Pytorch, you need to follow the API (i.e. for classification store your dataset in folders)
> - using a pretrained model and modifying it to adapt it to a similar task is easy. 
> - if you do not understand why we take this loss, that's fine, we'll cover that in Module 3.
> - even with a GPU, avoid unnecessary computations!

> To prepare next course:
Please have a look at the following videos and corresponding notebooks: [Module 2a - PyTorch tensors](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/) and [Module 2b - Automatic differentiation](https://dataflowr.github.io/website/modules/2b-automatic-differentiation/)

:sunflower:Session:two:
> 
## Content

- [**Module 1: Introduction & General Overview**](https://dataflowr.github.io/website/modules/1-intro-general-overview/) 
    - Intro: finetuning VGG for dogs vs cats [01_intro.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/01_intro.ipynb)
    - Practical: Using CNN for more dogs and cats [01_practical_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module1/01_practical_empty.ipynb)
- [**Module 2: Pytorch tensors and automatic differentiation**](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)
    - Basics on PyTorch tensors and automatic differentiation [02a_basics.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02a_basics.ipynb)
    - Linear regression from numpy to pytorch [02b_linear_reg.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02b_linear_reg.ipynb)
    - Practical: implementing backprop from scratch [02_backprop.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/02_backprop.ipynb)
    - Bonus: intro to JAX: autodiff the functional way [autodiff_functional_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_empty.ipynb) and its solution [autodiff_functional_sol.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/autodiff_functional_sol.ipynb)
    - Bonus: Linear regression in JAX [linear_regression_jax.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/linear_regression_jax.ipynb)
    - Bonus: automatic differentiation with dual numbers [AD_with_dual_numbers_Julia.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb)
- [**Homework 1: MLP from scratch**](https://dataflowr.github.io/website/homework/1-mlp-from-scratch/)
    - [hw1_mlp.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW1/hw1_mlp.ipynb)
- [**Module 3: Loss functions for classification**](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/)
    - An explanation of underfitting and overfitting with polynomial regression [03_polynomial_regression.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module3/03_polynomial_regression.ipynb)
- [**Module 4: Optimization for deep leaning**](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)
    - Practical: code Adagrad, RMSProp, Adam, AMSGrad [04_gradient_descent_optimization_algorithms_empty.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module4/04_gradient_descent_optimization_algorithms_empty.ipynb)
- [**Module 5: Stacking layers**](https://dataflowr.github.io/website/modules/5-stacking-layers/)
    - Practical: overfitting a MLP on CIFAR10 [Stacking_layers_MLP_CIFAR10.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module5/Stacking_layers_MLP_CIFAR10.ipynb)
- [**Module 6: Convolutional neural network**](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/)
    - Practical: build a simple digit recognizer with CNN [06_convolution_digit_recognizer.ipynb](https://github.com/dataflowr/notebooks/blob/master/Module6/06_convolution_digit_recognizer.ipynb)
- [**Homework 2: Class Activation Map and adversarial examples**](https://dataflowr.github.io/website/homework/2-CAM-adversarial/)
    - [HW2_CAM_Adversarial.ipynb](https://github.com/dataflowr/notebooks/blob/master/HW2/HW2_CAM_Adversarial.ipynb)
- [**Module 7: Dataloading**](https://dataflowr.github.io/website/modules/7-dataloading/)
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


Archives are available on the archive-2020 branch.

## Usage

If you want to run locally:

    python3 -m venv dldiy  # If you want to install packages in a virtualenv
    source dldiy/bin/activate

    pip install -r requirements.txt  # Install PyTorch and others
    python -m notebook  # Run notebook environment
