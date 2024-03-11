# ImagiNet: A Balanced High-Resolution Dataset for Benchmarking of Synthetic Image Detectors
![](media/dataset_preview.png)
**Abstract:**
<p align="justify">Generative models, such as Diffusion Models (DMs), Variational    Auto-Encoders (VAEs), and Generative Adversarial Networks (GANs),    produce images with a level of authenticity that makes them nearly    indistinguishable from real photos and artwork. While this capability    is beneficial for many industries, the difficulty of identifying    synthetic images leaves online media platforms vulnerable to    impersonation and misinformation attempts. To support the development    of defensive methods, we introduce ImagiNet, a high-resolution and    balanced dataset for synthetic image detection. It contains 200K    examples, spanning four content categories: photos, paintings, faces,    and uncategorized. Synthetic images are produced with open-source and    proprietary generators, whereas real counterparts of the same content    type are collected from public datasets. The structure of ImagiNet    allows for a two-track evaluation system: i) classification as real    or synthetic and ii) identification of the generative model. To    establish a baseline, we train a ResNet-50 model using a    self-supervised contrastive (SelfCon) objective for each track. The    model demonstrates strong performance across previously available    benchmarks, achieving an AUC of up to 0.97 and balanced accuracy    ranging between 80% and 93%, even under social network conditions    involving compression and resizing. </p>

## Instructions for downloading
For now we have provided only automatic download script for **FFHQ** datasetsince we used custom filenames. Before download, you should write the path to your credentials for Google Drive API in `ffhq.py` script and install the packages from `requirements.txt`. Other datasets should be downloaded the datasets from their original sources and place them in one directory. This is the root directory of the dataset. After that, you can invoke `delete_not_needed.py` with an argument to the path of the root directory where all datasets are extracted. The script will remove the images that are not included in our dataset.
> [!NOTE]
> You should place all extracted datasets in one folder.

The extracted content should be put in the following folders (<sup>*</sup> signifies datasets that do not require additional folder to be created):
 - **ImageNet:** imagenet (ILSVRC folder placed here)
 - **COCO<sup>*</sup>:** train2017, val2017
 - **LSUN/ProGAN<sup>*</sup>:** progan_train, progan_val
 - **WikiArt<sup>*</sup>:** wikiart
 - **LSUN/ProGAN<sup>*</sup>:** progan_train, progan_val
 - **Photozilla:** photozilla (dataset and other_dataset placed here)
 - **Danbooru:** danbooru2021
 - **JourneyDB:** journeydb (extracted 000.zip folder renamed to journeydb)

> [!WARNING]
> Currently, Danbooru host is down. Due to this, we provided a back up of the images [here]().


<!-- To download all the sets automatically, you will need to download the pip packages in `requirements.txt`. After that all the scripts should be invoked in the same directory. 
> [!NOTE]
> You should enter your Kaggle API and Google Drive API credentials in the scripts to download the images automatically. 


After downloading all the sets with the scripts, you can invoke `delete_not_needed.py`, to clear all the images in the directory, not used in the dataset.
-->


## Sources
The images generated in our work and DALL·E 3 images are provided [here](https://drive.google.com/drive/folders/1OGVxMuKKaRFwEJjhE_sX2ysB7nmn74M2?usp=drive_link).
The following sets are not provided in our archive. They should be downloaded from their original sources:
 - **ImageNet:** [Training/Test set](https://www.kaggle.com/c/imagenet-object-localization-challenge)
 - **COCO:** [Training set](http://images.cocodataset.org/zips/train2017.zip), [Validation set](http://images.cocodataset.org/zips/val2017.zip)
 - **LSUN/ProGAN**: [Training set](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view?usp=sharing), [Test set](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing)
- **FFHQ:** [Aligned images 1024x1024](https://drive.google.com/open?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)
- **WikiArt**: [Whole dataset](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk)
- **Danbooru:**: [Original images](https://gwern.net/danbooru2021)
- **Photozilla:**  [Original images](https://drive.google.com/file/d/1WkY6rcXMnir8nk4obejVq64h_WrtydVg/view?usp=drive_link)
- **JourneyDB:** [File 000.tgz](https://huggingface.co/datasets/JourneyDB/JourneyDB/blob/main/data/train/imgs/000.tgz)
## Licensing
- **Real Images:** Source images will adhere to their original licenses. Please respect the terms of use associated with any external image sources.
-  **Synthetic Images (DALL·E 3, Midjourney) :**
	- **DALL·E 3:** Part of the dataset images (folder dalle3) are sourced under **CC-0** license
	-  **Midjourney:** All images generated with Midjourney are gathered from **[JourneyDB](https://journeydb.github.io/)**, follow their license.
-   **Synthetic Images (SD v2.1, SDXL v1.0, Animagine XL, StyleGAN3, StyleGAN-XL,  DALL·E 3):** Images generated using these models are licensed under **CC BY 4.0**. This allows for:
	- **Part of DALL·E 3:** Images in folders **dalle3_additon** are generated by us and follow this license.
    -   **Sharing:** Free distribution and copying.
    -   **Adaptation:** Remixing, transforming, and building upon the material.
    -   **Commercial Use:** Utilization in commercial contexts.
    -   **Attribution:** Users must give appropriate credit, provide a link to the license, and indicate if changes were made.
## Further improvements
 - Automatic download scripts through Google Drive API and Kaggle API
