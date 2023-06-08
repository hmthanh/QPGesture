# QPGesture: Quantization-Based and Phase-Guided Motion Matching for Natural Speech-Driven Gesture Generation

## Abstract

Speech-driven gesture generation is challenging due to random jitters and the asynchronous relationship between speech and gestures.
A novel quantization-based and phase-guided motion matching framework is introduced.
Gesture VQ-VAE module learns a codebook to summarize gesture units, reducing random jittering.
Levenshtein distance is used to align gestures with speech, using audio quantization as a similarity metric.
Phase is introduced to guide optimal gesture matching based on the semantics of context or rhythm.
Extensive experiments demonstrate that the proposed method outperforms recent approaches in speech-driven gesture generation.

## 1. Introduction

## 2. Related Work

### End-to-end Co-speech Gesture Generation

### Quantization-based Pose representation


## 3. Our Approach


* Approach takes audio, text, seed pose, and optional control signals as inputs.
* Inputs and motion database are preprocessed into quantized discrete forms.
* Best candidate for speech and text are found separately.
* Optimal gestures are selected based on the phase corresponding to the seed code and the two candidates.


### 3.1. Learning a discrete latent space representation

#### Gesture Quantization

$$
x^2
$$

#### Audio Quantization

### 3.2. Motion Matching based on Audio and Text

#### Audio-based Search

#### Text-based Search

### 3.3. Phase-Guided Gesture Generation

## 4. Experiments

#### Implementation Details


#### Evaluation Metrics

### 4.1. Comparison to Existing Methods

#### User Study

### 4.2. Ablation Studies

#### User Study

### 4.3. Controllability

## 5. Discussion and Conclusion

* Paper presents a quantization-based and phase-guided motion matching framework for speech-driven gesture generation.
* Addresses the random jittering issue by using discrete representation for human gestures.
* Deals with the asynchronicity of speech and gestures and the flexibility of current motion matching models using Levenshtein distance based on audio quantization.
* Phase-guided audio-based or text-based candidates are used as the final result.
* Experiments on the BEAT dataset and user studies demonstrate that the proposed framework achieves state-of-the-art performance both qualitatively and quantitatively.
* Suggests the potential for improvement by considering additional modalities such as emotions and facial expressions to generate more appropriate gestures.


