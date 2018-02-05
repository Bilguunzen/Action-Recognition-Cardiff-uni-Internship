# Summer internship at Cardiff University (Adverserial Examples) 

## Background

## Changed

Adverserial images, have been largely discussed within the context of decieving
image classification model. Generating a perceptuallly identifiable image that
will confuse the trained CNN, brief introduction to the areas can be found
(here)[https://blog.openai.com/adversarial-example-research/]. But the general
concept can be understoood form this image:

![image](https://blog.openai.com/content/images/2017/02/adversarial_img_1.png)

To my knowledge there hasn't been any exploration of attacks on Video based
network, particularly on two-stream style networks.

## Hypothesis

Adverserial images have proved valid techniques, we hypothesise that due to the
nature of video recognition CNNs consisting of several methods of input they
would be less succeptable to such attacks. We intend to validate this through
the application of several attacks to all possible combinations of input
streams, and evaluating the effectivness of each attack.

## Key Milestones

[ ] Get a two stream network up and running

[ ] Read around possible attacks / Implementations in foolbox

[ ] List experiments for differing streams of the networks

[ ] Carry out experiments

[ ] Review and update hypothesis with results in hand


## Key Papers

Intriguing properties of neural networks, ICLR 2014

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan,
Ian Goodfellow, Rob Fergus
[arXiv](https://arxiv.org/abs/1312.6199)


Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine 
learning models, unpublished

Jonas Rauber, Wieland Brendel, Matthias Bethge
[arXiv](https://arxiv.org/abs/1707.04131) 


Two-Stream Convolutional Networks for Action Recognition in Videos, NIPS 2014

Karen Simonyan, Andrew Zisserman
[arXiv](https://arxiv.org/abs/1406.2199) 


Practical Black-Box Attacks against Machine Learning, ASIA CCS 2017

Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay
Celik, Ananthram Swami
[arXiv](https://arxiv.org/abs/1602.02697) 

Adverserial Examples In The Physical World, ICLR Workshop 2017

Alexey Kurakin, Ian J. Goodfellow, Samy Bengio
[arXiv](https://arxiv.org/abs/1607.02533) 

Two Stream Model files and Weights - [zisserman website](http://www.robots.ox.ac.uk/~vgg/software/two_stream_action/)
