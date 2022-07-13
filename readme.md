This repository got inspiration from repositories [probabilistic_unet](https://github.com/SimonKohl/probabilistic_unet) and (KiU-Net-pytorch)[https://github.com/jeya-maria-jose/KiU-Net-pytorch]. The main architecture are the same as Probabilistic U-Net but on the U-Net part we added Cross Residual Fusion Block (CRFB). CRFB function are for up-sampling the input image so the segmentation will increase the size too and help with segmentation result. Why we only add one CRFB because the resource limitation. Furthermore, it will be great if using KiU-Net as complete architecture to change U-Net from Probabilistic U-Net's architecture. Please check their original papers ("[A Probabilistic U-Net for Segmentation of Ambiguous Images](https://arxiv.org/abs/1806.05034)" and "[KiU-Net: Towards Accurate Segmentation of Biomedical Images using Over-complete Representations](https://arxiv.org/abs/2006.04878)") for more details information.
Training process:<br>
![Training](https://github.com/rizalmaulanaa/Prob-U-Net-CRFB/tree/master/Models-Probabilistic%20U-Net_CRFB_traning.png)<br>
Sampling process:<br>
![Sampling](https://github.com/rizalmaulanaa/Prob-U-Net-CRFB/tree/master/Models-Probabilistic%20U-Net_CRFB_sampling.png)<br>