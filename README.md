# :stars: STAR: End-to-End Speech-to-Audio Generation via Speech Semantic Representation Bridging
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://AnonymousGit0.github.io/STAR)

This work presents **STAR**, the first end-to-end speech-to-audio generation framework, designed to enhance efficiency and address error propagation inherent in cascaded systems. 
It:
* Recognize the potential of the speech-to-audio generation task and have designed the first E2E system STAR;
* Validate E2E STA feasibility via representation learning experiments, showing that spoken sound event semantics can be directly extracted;
* Achieve effective speech-to-audio modal alignment through a bridge network mapping mechanism and a two-stage training strategy;
* Significantly reduces speech processing latency from 156ms to 36ms(≈ 76.9% reduction), whilesurpassing the generation performance of cascaded systems.

### Table of Contents
 - [Environment and Data Preparation](#Preparation)
 - [Stage1: Bridge Network](#Bridge)
 - [Stage2: STA Generation](#STA)

***


<a id="Preparation"></a>
### :scissors: Data Preparation

Generating corresponding speech from captions in Audiocaps, followed by feature extraction using different speech encoders (DAC, Hubert, WavLM):
```shell
git clone https://github.com/AnonymousGit0/STAR
python src/data_preparation/vits/vits_inference.py
python src/data_preparation/data_preparation/speech_encoder/hubert_extract_feature.py
```

***

<a id="Bridge"></a>
### :bulb: Stage1: Bridge Network
Pre-train the Bridge Network using sound event labels from AudioSet
```shell
python src/bridge_network/qformer_predictions.py
```

***

<a id="STA"></a>
### :seedling: Stage2: STA Generation
Train end-to-end speech-to-audio generation using speech-audio data
 ```shell
sh src/sta_generation/bash_scripts_star/train_star_fm.sh
sh src/sta_generation/bash_scripts_star/infer_multi_gpu.sh
python src/sta_generation/evaluation/star.py --gen_audio_dir {generated_audio_folder}
```

## Acknowledgement
Our code referred to the [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm), [fairseq](https://github.com/facebookresearch/fairseq), [DAC](https://github.com/descriptinc/descript-audio-codec), [SECap](https://github.com/thuhcsi/SECap), [HEAR](https://hearbenchmark.com/). We appreciate their open-sourcing of their code.





