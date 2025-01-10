# TD4V: Temporal Difference Module for Efficient Video Action Recognition via Fine-tuning and Side-tuning
Official implementation of TD4V for Video Action Recognition
We will upload it in its entirety as soon as possible
The core data of TD4V is at clip/adapter.py
# Abstract
Parameter-Efficient Transfer Learning (PETL) in video action recognition (VAR) effectively mitigates the challenges of transfer learning and facilitates the utilization of reliable prior knowledge from visual-language models. Recently, side-tuning methods have emerged in VAR, significantly reducing memory usage without excessively increasing tunable parameters. However, most existing PETL works in VAR focus solely on either "parameter efficiency" or "memory efficiency." In this paper, we propose a plug-and-play module named Temporal Difference Module for Video Action Recognition (TD4V), which achieves superior performance in both fine-tuning for "parameter efficiency" and side-tuning for "memory efficiency." TD4V explicitly extracts temporal clues through difference operations and utilizes a 2D convolution layer to extract features from the temporal difference information distributed in space. We integrate TD4V into CLIP for both fine-tuning and side-tuning scenarios. In fine-tuning, we freeze all parameters of CLIP and only tune TD4V with minimal parameters. In side-tuning, we design a lightweight CLIP with TD4V as the side network, significantly reducing memory usage by 60.7% (from 34.9G in fine-tuning to 13.7G in side-tuning) through backpropagation that passes only through the lightweight side network. Experimental results demonstrate that TD4V achieves state-of-the-art or comparable performance on benchmark datasets (80.37% on HMDB-51, 96.91% on UCF-101, and 70.18% on SSv2) with significantly fewer parameters (2.14M in fine-tuning and 20.44M in side-tuning) compared to existing methods. Our approach thus offers a practical solution for efficient video action recognition through PETL.
# Requirement
- PyTorch >= 1.8
- RandAugment
- pprint
- dotmap
- yaml
# Structure Description
|-clip  // our main code
 |-adapter.py  // where the core data of our TD4V at
 |-...  // other main code
|-configs  // where can change experimental settings
 |-hmdb51.yaml  // hmdb51 and ucf101
 |-ssv2.yaml  // ssv2
|-scripts
 |-run_train_vision.sh  // run the code by sh
|-train_vision.py  // here can change other settings
|-ReadMe.md
|-...
