# StemgenRT training assets

`stemgenrt-0.8.0.tar.gz` contains the [upstream StemgenRT source](https://github.com/sweetspotsoundsystem/stemgen-rt) and its MIT license. StemgenRT is maintained by Axel Delafosse. The source license credits Phil Wang and Axel Delafosse.

`vocals_released.pt` is a PyTorch conversion of the [released StemgenRT vocals ONNX model](https://github.com/sweetspotsoundsystem/stemgen-rt/blob/main/model/parallel/vocals.onnx). `onnx_to_torch.py` maps the ONNX initializers back into a PyTorch state dictionary for fine-tuning. The source ONNX has SHA-256 `f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148`.

The converted weights are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [the upstream weights notice](WEIGHTS-LICENSE.md) and `MODEL_CARD.md` in the source archive. The original model was trained on [MUSDB18-HQ](https://zenodo.org/records/3338373) by Z. Rafii et al. and [MoisesDB](https://arxiv.org/abs/2307.15913) by I. Pereira, F. Araújo, F. Korzeniowski and R. Vogl.
