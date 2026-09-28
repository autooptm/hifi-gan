<div align="center">
  <a href="https://autooptm.com"><img src=".autooptm/logo.png" width="96" alt="AutoOptm"></a>

  <h1>hifi-gan · optimized by <a href="https://autooptm.com">AutoOptm</a></h1>

  <p><b>1.47x faster end to end</b> on the command below, output verified against the stock program.</p>

  <p>
    <a href="https://autooptm.com"><img alt="speedup" src="https://img.shields.io/badge/end--to--end-1.47x-2ea44f"></a>
    <a href="https://github.com/jik876/hifi-gan/commit/4769534d45265d52a904b850da5a622601885777"><img alt="base" src="https://img.shields.io/badge/upstream-4769534d4526-blue"></a>
    <img alt="card" src="https://img.shields.io/badge/measured%20on-RTX%204090-lightgrey">
  </p>
</div>

> This is a fork of [jik876/hifi-gan](https://github.com/jik876/hifi-gan) at commit
> [`4769534d4526`](https://github.com/jik876/hifi-gan/commit/4769534d45265d52a904b850da5a622601885777) with the AutoOptm patch applied on top.
> The optimisation was found, measured and verified automatically by [AutoOptm](https://autooptm.com);
> the patch is also kept verbatim at [`.autooptm/autooptm.patch`](.autooptm/autooptm.patch).

Every change is on by default and is behind a switch; see [`.autooptm/autooptm.patch`](.autooptm/autooptm.patch). `meldataset.py` also carries the two call-site fixes that current librosa (>= 0.10) and PyTorch (>= 2.0) require; the arithmetic there is unchanged.

## The result

| | |
|---|---|
| **Command** | `python train.py --config config_v1.json` |
| **Entry point** | `train.py` |
| **Unit measured** | one training step on LJSpeech (generator forward → MPD + MSD discriminator update → generator update), batch 16 × 8192 samples |
| **Before (stock)** | 189.7 ms per unit |
| **After (this tree, all switches at their defaults)** | 129.0 ms per unit |
| **Speedup** | **1.47x** end to end on RTX 4090, wall clock over two full epochs (1,615 steps, data loading included), run-to-run spread 0.23% |
| **Output** | per-step loss curve interleaves the stock curve (mean gap 0.66% of the loss); mel-spectrogram L1 after two epochs 0.544 / 0.539 against stock's 0.545 / 0.544 |

### What changed

| File | Where | Gain (alone) |
|---|---|---|
| `train.py` | train() | 1.106x |
| `models.py` | MultiPeriodDiscriminator.forward / MultiScaleDiscriminator.forward | 1.084x |
| `models.py` | Generator.forward | 1.061x |
| `models.py` | discriminator_loss() | 1.0x |
| `meldataset.py` | mel_spectrogram() | 1.0x, needed on current librosa / PyTorch |

## Reproduce

```bash
git clone https://github.com/autooptm/hifi-gan-ao.git
cd hifi-gan-ao
# set up exactly as upstream documents (LJSpeech wavs in LJSpeech-1.1/wavs), then:
python train.py --config config_v1.json
```

The diff against upstream is one commit: `git log -1 -p` shows it, and
`git diff 4769534d4526` is the same patch as `.autooptm/autooptm.patch`.

---

<div align="center"><sub>Optimized by <a href="https://autooptm.com">AutoOptm</a> — point it at a repository, get back a verified speedup and the patch.</sub></div>

---

# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

### Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

In our [paper](https://arxiv.org/abs/2010.05646), 
we proposed HiFi-GAN: a GAN-based model capable of generating high fidelity speech efficiently.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
Several recent work on speech synthesis have employed generative adversarial networks (GANs) to produce raw waveforms. 
Although such methods improve the sampling efficiency and memory usage, 
their sample quality has not yet reached that of autoregressive and flow-based generative models. 
In this work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. 
As speech audio consists of sinusoidal signals with various periods, 
we demonstrate that modeling periodic patterns of an audio is crucial for enhancing sample quality. 
A subjective human evaluation (mean opinion score, MOS) of a single speaker dataset indicates that our proposed method 
demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times faster than 
real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the mel-spectrogram inversion of unseen 
speakers and end-to-end speech synthesis. Finally, a small footprint version of HiFi-GAN generates samples 13.4 times 
faster than real-time on CPU with comparable quality to an autoregressive counterpart.

Visit our [demo website](https://jik876.github.io/hifi-gan-demo/) for audio samples.


## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
python train.py --config config_v1.json
```
To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.

Validation loss during training with V1 generator.<br>
![validation loss](./validation_loss.png)

## Pretrained Model
You can also use pretrained models we provide.<br/>
[Download pretrained models](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)<br/> 
Details of each folder are as in follows:

|Folder Name|Generator|Dataset|Fine-Tuned|
|------|---|---|---|
|LJ_V1|V1|LJSpeech|No|
|LJ_V2|V2|LJSpeech|No|
|LJ_V3|V3|LJSpeech|No|
|LJ_FT_T2_V1|V1|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|LJ_FT_T2_V2|V2|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|LJ_FT_T2_V3|V3|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|VCTK_V1|V1|VCTK|No|
|VCTK_V2|V2|VCTK|No|
|VCTK_V3|V3|VCTK|No|
|UNIVERSAL_V1|V1|Universal|No|

We provide the universal model with discriminator weights that can be used as a base for transfer learning to other datasets.

## Fine-Tuning
1. Generate mel-spectrograms in numpy format using [Tacotron2](https://github.com/NVIDIA/tacotron2) with teacher-forcing.<br/>
The file name of the generated mel-spectrogram should match the audio file and the extension should be `.npy`.<br/>
Example:
    ```
    Audio File : LJ001-0001.wav
    Mel-Spectrogram File : LJ001-0001.npy
    ```
2. Create `ft_dataset` folder and copy the generated mel-spectrogram files into it.<br/>
3. Run the following command.
    ```
    python train.py --fine_tuning True --config config_v1.json
    ```
    For other command line options, please refer to the training section.


## Inference from wav file
1. Make `test_files` directory and copy wav files into the directory.
2. Run the following command.
    ```
    python inference.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
    ```
    python inference_e2e.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option.


## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

