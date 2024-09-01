# Sample-Efficient Diffusion for Text-To-Speech Synthesis

This is the official code release for the Interspeech 2024 paper:

**Sample-Efficient Diffusion for Text-To-Speech Synthesis**.

by Justin Lovelace, Soham Ray, Kwangyoun Kim, Kilian Q. Weinberger, and Felix Wu

**Note: Code and model checkpoint will be available soon. Stay tuned for updates!**

### Abstract
This work introduces Sample-Sample-Efficient Speech Diffusion (SESD), an algorithm for effective speech synthesis in modest data regimes through latent diffusion. It is based on a novel diffusion architecture, that we call U-Audio Transformer (U-AT), that efficiently scales to long sequences and operates in the latent space of a pre-trained audio autoencoder. Conditioned on character-aware language model representations, SESD  achieves impressive results despite training on less than 1k hours of speech – far less than current state-of-the-art systems. In fact, it synthesizes more intelligible speech than the state-of-the-art auto-regressive model, VALL-E, while using less than 2% the training data.
### Citation
```bibtex
@inproceedings{lovelace2024sesd,
  title={Sample-Efficient Diffusion for Text-To-Speech Synthesis},
  author={Lovelace, Justin and Ray, Soham and Kim, Kwangyoun and Weinberger, Kilian Q. and Wu, Felix},
  booktitle={Interspeech 2024},
  year={2024},
  publisher={ISCA},
}
