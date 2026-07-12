# CS231n Course Backbone

This summary is based on the three CS231n-style assignments adapted in this
repository. The course arc is not just a list of model families; it is a
progression from low-level numerical building blocks to modern visual
representation learning and generative modeling.

## 1. Classical Image Classification

The first layer of the course is the supervised image-classification problem:
turn an image into a class label and measure success with train/validation/test
accuracy. Assignment 1 starts with CIFAR-10 and deliberately simple models so
the core mechanics are visible.

- k-nearest neighbors introduces data representation, distance metrics, and the
  basic train/test split.
- Linear classifiers introduce scores, losses, regularization, and gradient
  descent.
- Softmax and SVM-style objectives show how a classifier turns raw scores into
  optimization problems.
- Feature pipelines such as HOG and color histograms show the pre-deep-learning
  approach: design features first, then train a shallow classifier.

The backbone idea is that machine learning starts as an optimization problem
over data, labels, scores, losses, and parameters.

## 2. Neural Network Primitives

The next layer is implementing neural networks from first principles. The
course makes the forward pass, backward pass, and parameter update explicit
before relying on high-level frameworks.

- Affine layers, ReLU, normalization, dropout, convolution, pooling, and loss
  layers define the reusable components.
- Backpropagation connects local derivatives into full model gradients.
- Gradient checking provides a practical sanity check for manual
  implementations.
- Optimizers such as SGD, momentum, RMSProp, and Adam show that training
  dynamics are part of the model-building toolkit.

The backbone idea is that deep learning is modular differentiable programming:
compose layers, compute gradients, update parameters, and verify every part.

## 3. Training Deep Networks Reliably

Assignment 2 shifts from basic networks to techniques that make deeper models
trainable and useful.

- Batch normalization and layer normalization stabilize activation statistics.
- Dropout regularizes models by injecting noise during training.
- Convolutional networks exploit image locality and weight sharing.
- PyTorch introduces automatic differentiation and framework-level model
  construction while preserving the concepts learned from scratch.

The backbone idea is that model architecture and training procedure are coupled:
normalization, regularization, initialization, optimization, and framework
abstractions all affect whether a network actually trains.

## 4. Visual Recognition Architectures

The assignments move from fully connected networks to architectures matched to
visual data.

- CNNs encode spatial inductive bias through local filters, pooling, and
  hierarchical features.
- Vision Transformers split images into patches and use attention instead of
  convolution as the main mixing operation.
- The course compares hand-designed visual bias with more general
  sequence-modeling machinery.

The backbone idea is that representation architecture matters: the way a model
mixes pixels into features determines what it can learn efficiently.

## 5. Sequence Modeling and Captioning

The course then connects vision to language through image captioning.

- Vanilla RNN captioning models generate word sequences from image features.
- Transformer captioning replaces recurrence with attention-based sequence
  modeling.
- Captioning exposes the encoder-decoder pattern: extract visual features, then
  decode structured output.

The backbone idea is that computer vision is not only classification; visual
features can condition sequential and multimodal predictions.

## 6. Attention and Transformers

Assignment 3 makes attention a central primitive.

- Multi-head attention learns content-dependent interactions between tokens.
- Positional encoding restores order information missing from pure attention.
- Transformer decoders support caption generation.
- Transformer encoders support Vision Transformer classification.

The backbone idea is that attention is a general mechanism for routing
information between parts of an input or output, and it can replace recurrence
or convolution in many settings.

## 7. Self-Supervised and Multimodal Representation Learning

The later material shifts from supervised labels to representation learning.

- SimCLR uses contrastive learning to train image encoders without manual
  labels.
- CLIP aligns image and text representations in a shared embedding space,
  enabling retrieval and zero-shot classification.
- DINO shows that self-supervised vision transformers can learn features useful
  for dense visual tasks such as segmentation.

The backbone idea is that large-scale representation learning can produce
general-purpose visual features, sometimes reducing the need for task-specific
labels.

## 8. Generative Modeling

The final layer introduces diffusion models.

- DDPMs define a forward noising process and learn the reverse denoising
  process.
- UNet-style architectures predict noise or clean images at each diffusion
  timestep.
- Text conditioning and classifier-free guidance connect generation to prompts.

The backbone idea is that modern vision models are not limited to predicting
labels; they can learn image distributions and synthesize new visual data.

## Overall Progression

The three-assignment sequence can be read as one ladder:

1. Start with image classification and explicit numerical optimization.
2. Build neural-network layers, gradients, and optimizers by hand.
3. Add practical training tools such as normalization and dropout.
4. Scale from fully connected networks to CNNs and PyTorch models.
5. Connect vision to language with RNNs and Transformers.
6. Generalize attention to both text-like sequences and image patches.
7. Learn representations from unlabeled or multimodal data.
8. Move from recognition to generation with diffusion models.

The durable course backbone is therefore:

> visual data -> features -> scores/losses -> gradients -> trainable networks
> -> architecture design -> sequence and attention models -> representation
> learning -> generative modeling.
