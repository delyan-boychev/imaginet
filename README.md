<p align="center">
    <img src="media/logo.webp" alt="" width="30%" align="top"">
</p>

<p align="center">
  <em><strong>ImagiNet</strong>: A Multi-Content Dataset for Generalizable Synthetic Image Detection via Contrastive Learning <br></em>
</p>

**Abstract:**
<p align="justify">Generative models, such as diffusion models (DMs), variational autoencoders (VAEs), and generative adversarial networks (GANs), produce images with a level of authenticity that makes them nearly indistinguishable from real photos and artwork. While this capability is beneficial for many industries, the difficulty of identifying synthetic images leaves online media platforms vulnerable to impersonation and misinformation attempts. To support the development of defensive methods, we introduce ImagiNet, a high-resolution and balanced dataset for synthetic image detection, designed to mitigate potential biases in existing resources. It contains 200K examples, spanning four content categories: photos, paintings, faces, and uncategorized. Synthetic images are produced with open-source and proprietary generators, whereas real counterparts of the same content type are collected from public datasets. The structure of ImagiNet allows for a two-track evaluation system: i) classification as real or synthetic and ii) identification of the generative model. To establish a baseline, we train a ResNet-50 model using a self-supervised contrastive objective (SelfCon) for each track. The model demonstrates state-of-the-art performance and high inference speed across established benchmarks, achieving an AUC of up to 0.99 and balanced accuracy ranging from 86% to 95%, even under social network conditions that involve compression and resizing.</p>

![](media/dataset_preview.png)

## Instructions for downloading
For now we have provided only an automatic download script for **FFHQ** datasets since we used custom filenames. Before downloading, you should write the path to your credentials for Google Drive API in `download_scripts/ffhq.py` script and install the packages from `requirements.txt`. Other datasets should be downloaded the datasets from their original sources and placed in one directory. This is the root directory of the dataset. After that, you can invoke `dataset_operations/delete_not_needed.py` with an argument to the path of the root directory where all datasets are extracted. The script will remove the images that are not included in our dataset.
| :information_source: You should place all extracted datasets in one folder. |
|-----------------------------------------------------------------------------|

The extracted content should be put in the following folders (* signifies datasets that do not require additional folders to be created):
 - **ImageNet:** imagenet (ILSVRC folder placed here)
 - **COCO\*:** train2017, val2017
 - **LSUN/ProGAN\*:** progan_train, progan_val
 - **WikiArt\*:** wikiart
 - **LSUN/ProGAN\*:** progan_train, progan_val
 - **Photozilla:** photozilla (dataset and other_dataset placed here)
 - **Danbooru:** danbooru2021
 - **JourneyDB:** journeydb (extracted 000.zip folder renamed to journeydb)

The annotations for the test, train, and calibration sets as well as the annotations for model group testing are provided in `annotations` folder.
| :warning:  Currently, Danbooru host is down. Due to this, we provided a backup of the images [here](https://drive.google.com/file/d/1p0EM6IUAdBhFfdGoLpo0ewhLPXEkA86a/view?usp=sharing).  |
|-------------------------------------------------------------------------------------------------------------|


<!-- To download all the sets automatically, you will need to download the pip packages in `requirements.txt`. After that all the scripts should be invoked in the same directory. 
> [!NOTE]
> You should enter your Kaggle API and Google Drive API credentials in the scripts to download the images automatically. 


After downloading all the sets with the scripts, you can invoke `delete_not_needed.py`, to clear all the images in the directory, not used in the dataset.
-->
## Training and testing sets
All annotations for the training and testing sets are provided in the annotations folder. To achieve the perturbed set for testing, you should run `dataset_operations/save_testset.py`. It will preprocess the needed images for testing. All the scripts for testing are provided in `testing_utils` folder.
| :information_source:  You should download all testsets (Corvi, Practical testset) to run the scripts. We provide only the testing annotations. The Corvi test set should be preprocessed with a resolution of 256 x 256. |
|-------------------------------------------------------------------------------------------------------------|

Before testing you should run `testing_utils/testing/setup_testing.py` to download the needed weights and libraries. All scripts are reproducible. 

We also provide our training scripts in `training_utils`. You can train the model with either SelfCon or Cross-Entropy Loss. The calibration scripts are placed in the same directory.

## Model Checkpoints
Model checkpoints are accessible [here](https://drive.google.com/drive/folders/1En2BI9H9LxqA5XIpNaMXhqhF8--XAKns?usp=sharing).


## Sources
The images generated in our work and DALL·E 3 images are provided [here](https://drive.google.com/file/d/1uUAoVUcAlUX9ltOXBlf3pKDz3rCKoytk/view?usp=sharing).
The following sets are not provided in our archive. They should be downloaded from their original sources:
 - **ImageNet:** [Training/Test set](https://www.kaggle.com/c/imagenet-object-localization-challenge)
 - **COCO:** [Training set](http://images.cocodataset.org/zips/train2017.zip), [Validation set](http://images.cocodataset.org/zips/val2017.zip)
 - **LSUN/ProGAN:** [Training set](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view?usp=sharing), [Test set](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing)
- **FFHQ:** [Aligned images 1024x1024](https://drive.google.com/open?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)
- **WikiArt:** [Whole dataset](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk)
- **Danbooru:** [Original images](https://gwern.net/danbooru2021)
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
