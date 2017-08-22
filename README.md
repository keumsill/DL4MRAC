# DL4MRAC
Deep Learning for MR-based attenuation correction

[Background of the project]
Simultaneous PET/MR imaging systems have shown promise to provide simultaneous molecular and morphological characterization of a variety of diseases [1]. However, challenges in MR-based attenuation correction (MRAC) methods in certain parts of the body are still complicating the goal of quantitative PET/MR. These problems are due to the difficulty of imaging and segmenting bone with current MR sequences [2]. With currently available attenuation maps for whole body PET/MR, bone segmentation and classification using MR images are sub-optimal, especially at boundaries with air or soft tissue, causing quantification errors when using PET/MR with MRAC for evaluating disease progression and treatment response [3]. We previously implemented deep learning techniques with a convolutional Auto-encoder, an artificial neural network used for unsupervised learning of efficient codings, using 2-point Dixon sequence MR-images to show feasibility [4]. In this work, we automatically delineate bone directly from zero-TE (ZTE) MR images by deploying a Convolutional Neural Network, based on the TensorFlow library [5][6]. This can be added onto the currently available bone-missing whole-body MRAC map. We found that the training cost in a majority of voxels was reduced by 36.5 % with an epoch size of 100. This study shows the potential of using deep learning techniques and convolutional neural networks to generate bone-included attenuation maps for PET quantification.

[Deep Learning]
A series of MRI-ZTE images were segmented using Seg3D segmentation software [7] to highlight different structures in the body. The ZTE-MR image and segmented bone images were exported as gray scale JPEG images with matrix size of 110x110 and pixel size of 2.4x2.4. 116 slices of axial ZTE-MR images were used as the training data (network input) for the Auto-encoder network and 116 slices of axial bone-segmented ZTE images were set as the target data (network label). The network uses four 3x3 convolutional layers to encode the input image into an inner-most latent layer. Then four convolutional layers were used to decode this latent layer and create the 110x110 image output. 
TensorFlow has Adam optimization built in [8], and this method was used for optimizing our weights during training.


[References]
[1] D. A. Torigian, H. Zaidi, T. C. Kwee, B. Saboury, J. K. Udupa, Z.-H. Cho, and A. Alavi, “PET/MR imaging: technical aspects and potential clinical applications,” Radiology, vol. 267, no. 1, pp. 26–44, 2013.
[2] M. Hofmann, I. Bezrukov, F. Mantlik, P. Aschoff, F. Steinke, T. Beyer, B. J. Pichler, and B. Scho ̈lkopf, “MRI-based attenuation correction for whole-body PET/MRI: quantitative evaluation of segmentation-and atlas-based methods,” Journal of Nuclear Medicine, vol. 52, no. 9, pp. 1392–1399, 2011.
[3] K. S. Lee, G. Zaharchuk, P. K. Gulaka, and C. S. Levin, “Evaluation of zero-TE-based attenuation correction methods on PET quantification of PET/MRI head and neck lesions,” Proceedings, IEEE Nuclear Science Symposium and Medical Imaging Conference, 2016.
[4] L. Tao, K. S. Lee, and C. S. Levin, “Study of a Convolutional Autoencoder for Automatic Generation of MR-based Attenuation Map in PET/MR,” IEEE MIC2017, accepted for presentation.
[5] P. K. Mital, “Tensorflow tutorials,” https://github.com/pkmital/tensorflow tutorials, accessed: 2017-05-05.
[6] Y. Bengio et al., “Learning deep architectures for AI,” Foundations and trends in Machine Learning, vol. 2, no. 1, pp. 1–127, 2009.
[7] CIBC, “Seg3D: Volumetric Image Segmentation and Visualization” Scientific Computing and Imaging Institute (SCI) (2016) http://www.seg3d.org [8] D. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.
