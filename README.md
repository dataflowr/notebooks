Material for Deep Learning hands-on courses.

Largely inspired by [fast.ai](http://www.fast.ai/) course: Practical Deep Learning For Coders but with a different focus.

The main goal of the courses is to allow students to understand papers, blog posts and codes available online and to adapt them to their projects as soon as possible. In particular, we avoid the use of any high-level neural networks API and focus on the [PyTorch](https://pytorch.org/) library in Python. 

[Constructive comments welcome!](#contact)

## two courses:

- [5 days course](#fasterai)

A 5 days crash course on deep learning for students with a background in python and ML.

- [Deep Learning: Do It Yourself!](https://www.di.ens.fr/~lelarge/dldiy/)

currently taught at ENS, under progress... (stay tuned for updates)

## main updates:

- 10/21/2018 ressources for [faster.ai](#fasterai) and [projects of the students](https://mlelarge.github.io/dataflowr/projects_xhec18.html) online. 

## [jobs](https://mlelarge.github.io/dataflowr/jobs.html)

## faster.ai

- Day 1:
  
  1. [introductory slides](https://mlelarge.github.io/dataflowr/Slides/01_intro_dldiy/index.html)
  2. first example: [dogs and cats with VGG](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/01_intro_DLDIY.ipynb)
  3. making a regression with autograd: [intro to pytorch](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/02_basics_pytorch.ipynb)
  4. using colab to compute features, just run the following [notebook](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/04_dogscast_features_colab.ipynb) on colab. Save it in your drive and open it with colab. It should load the features of dogs and cats in your drive.

- Day 2:
  
  5. understanding [convolutions](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/03_convolution-digit-recognizer_empty.ipynb) and your first neural network.
  6. pausing a bit, [slides recap](https://mlelarge.github.io/dataflowr/Slides/05_some_basics/index.html) on writing a module and BCE loss.
  7. using colab features to [overfit](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/04_dogscast_fromcolab_emty.ipynb) . This practical requires the features computed on colab in day 1.
  8. discovering embeddings with [collaborative filtering](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/05_collaborative_filtering.ipynb)

- Day 3:
  
  9. regularization, dropout, batchnorm, residual net, slides to come...
  10. introduction to [NLP](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/introduction_NLP.ipynb) with sentiment analysis.
  11. optimization, initialization, slides to come...
  12. unsupervised learning with [autoencoders](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/05_Autoencoder_empty.ipynb)

- Day 4:
  
  13. Recurrent Neural Networks, slides to come
  14. pytorch tutorial on [RNN](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/char_rnn_classification_tutorial.ipynb)
  14. understanding my network: [class activation map](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/CAM.ipynb)

- Day 5:
  
  15. Generative Adversarial Networks, [slides](https://mlelarge.github.io/dataflowr/Slides/GAN/index.html)
  16. [Conditional and info GANs](https://github.com/mlelarge/dataflowr/blob/master/Notebooks/GAN_double_moon_empty.ipynb)
  
<table align='center'>
<tr align='center'>
<td> GAN</td>
<td> InfoGAN</td>
</tr>
<tr>
  <td><img src = 'Slides/GAN/GAN_action.gif'/></td>
  <td><img src = 'Slides/GAN/IGAN_action.gif'/></td>
</tr>
</table>

### Contact

email: dataflowr@gmail.com

[homepage](https://www.di.ens.fr/~lelarge/#contact)
