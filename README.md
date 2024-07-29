<p align="center">
    <img src="media/logo.webp" alt="" width="40%" align="top"">
</p>

<p align="center">
  <em><strong>ImagiNet</strong>: A Multi-Content Dataset for Generalizable Synthetic Image Detection via Contrastive Learning <br></em>
</p>

![](media/dataset_preview.png)

<p align="justify">Generative models, such as diffusion models (DMs), variational autoencoders (VAEs), and generative adversarial networks (GANs), produce images with a level of authenticity that makes them nearly indistinguishable from real photos and artwork. While this capability is beneficial for many industries, the difficulty of identifying synthetic images leaves online media platforms vulnerable to impersonation and misinformation attempts. To support the development of defensive methods, we introduce ImagiNet, a high-resolution and balanced dataset for synthetic image detection, designed to mitigate potential biases in existing resources. It contains 200K examples, spanning four content categories: photos, paintings, faces, and uncategorized. Synthetic images are produced with open-source and proprietary generators, whereas real counterparts of the same content type are collected from public datasets. The structure of ImagiNet allows for a two-track evaluation system: i) classification as real or synthetic and ii) identification of the generative model. To establish a baseline, we train a ResNet-50 model using a self-supervised contrastive objective (SelfCon) for each track. The model demonstrates state-of-the-art performance and high inference speed across established benchmarks, achieving an AUC of up to 0.99 and balanced accuracy ranging from 86% to 95%, even under social network conditions that involve compression and resizing.</p>

## Dataset

A packed version of the dataset can be downloaded from [Huggingface](https://huggingface.co/datasets/delyanboychev/imaginet) manually or with their CLI.

```bash
huggingface-cli download delyanboychev/imaginet --repo-type dataset
```

To unzip the whole dataset, `7z` could be used as follows: 

```
7z x imaginet.7z.001 -oDIRECTORY
```

### Build from source

The dataset can be constructed from the original sources cited in our work in combination with the synthetic images we generate manually.

1. Install requirements with `pip install -r requirements.txt`.
2. Download all datasets from their original sources and place them in a directory.
 - The images generated in our work and DALL·E 3 images are provided [here](https://drive.google.com/file/d/1uUAoVUcAlUX9ltOXBlf3pKDz3rCKoytk/view?usp=sharing). Unzip the archive in the dataset directory.
 - **ImageNet:** [Training/Test set](https://www.kaggle.com/c/imagenet-object-localization-challenge). Put `imagenet` (ILSVRC folder) in the dataset directory.
 - **COCO:** [Training set](http://images.cocodataset.org/zips/train2017.zip), [Validation set](http://images.cocodataset.org/zips/val2017.zip). Put `train2017`, `val2017` in the dataset directory.
 - **LSUN/ProGAN:** [Training set](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view?usp=sharing), [Test set](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing). Put `progan_train`, `progan_val` in the dataset directory.
- **FFHQ:** [Aligned images 1024x1024](https://drive.google.com/open?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL). We have provided an automatic download script for **FFHQ** datasets since we used custom filenames. Write the path to your credentials for Google Drive API in `download_scripts/ffhq.py` and execute the script.
- **WikiArt:** [Whole dataset](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view?usp=drivesdk). Put `wikiart` in the dataset directory.
- **Danbooru:** [Original images](https://gwern.net/danbooru2021). Put `danbooru2021` in the dataset directory.
- **Photozilla:**  [Original images](https://drive.google.com/file/d/1WkY6rcXMnir8nk4obejVq64h_WrtydVg/view?usp=drive_link). Put `dataset`, `other_dataset` in the dataset directory.
- **JourneyDB:** [File 000.tgz](https://huggingface.co/datasets/JourneyDB/JourneyDB/blob/main/data/train/imgs/000.tgz). Put `journeydb` (extract `000.zip` folder & rename to `journeydb`) in the dataset directory.
3. Execute `python dataset_operations/delete_not_needed.py --path DIRECTORY` to extract the images included in our dataset and clean the rest.

| :warning:  Currently, Danbooru host is down. Due to this, we provided a backup of the images [here](https://drive.google.com/file/d/1p0EM6IUAdBhFfdGoLpo0ewhLPXEkA86a/view?usp=sharing).  |
|-------------------------------------------------------------------------------------------------------------|

## Models

Synthetic Image Detection results:

| ACC/AUC    | Grag2021        | Corvi2022       | Wu2023          | Corvi2022*      | Ours*           |
|------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| GAN        | 0.6889 / 0.8403 | 0.6822 / 0.8033 | 0.6508 / 0.6971 | 0.8534 / 0.9416 | 0.9372 / 0.9886 |
| SD         | 0.5140 / 0.5217 | 0.6112 / 0.6851 | 0.6367 / 0.6718 | 0.8693 / 0.9582 | 0.9608 / 0.9922 |
| Midjourney | 0.4958 / 0.5022 | 0.5826 / 0.6092 | 0.5326 / 0.5289 | 0.8880 / 0.9658 | 0.9652 / 0.9949 |
| DALL·E 3   | 0.4128 / 0.3905 | 0.5180 / 0.5270 | 0.5368 / 0.5482 | 0.8906 / 0.9759 | 0.9724 / 0.9963 |
| Mean       | 0.5279 / 0.5637 | 0.5985 / 0.6562 | 0.5892 / 0.6115 | 0.8753 / 0.9604 | 0.9589 / 0.9930 |

Model Identification results:

| Grag2021 | Wu2023 | Corvi2022 | Ours*  |
|----------|--------|-----------|--------|
| 24.30    | 16.01  | 49.53 	| 25.10  |

Checkpoints for models with * are accessible [here](https://drive.google.com/drive/folders/1En2BI9H9LxqA5XIpNaMXhqhF8--XAKns?usp=sharing).

## Training and Testing
All annotations for the training and testing sets are provided in the `annotations` folder. To achieve the perturbed set for testing, you should run `dataset_operations/save_testset.py`. It will preprocess the needed images for testing. All the scripts for testing are provided in `testing_utils` folder.
| :information_source:  You should download all testsets (Corvi, Practical testset) to run the scripts. We provide only the testing annotations. The Corvi test set should be preprocessed with a resolution of 256 x 256. |
|-------------------------------------------------------------------------------------------------------------|

Before testing you should run `testing_utils/testing/setup_testing.py` to download the needed weights and libraries. All scripts are reproducible. 

We also provide our training scripts in `training_utils`. You can train the model with either SelfCon or Cross-Entropy Loss. The calibration scripts are placed in the same directory.

## Cite

```
BIBTEX TBA
```

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

| :information_source: If the licensing of the referenced works has been updated or you have other complaints, please open an issue. |
|-------------------------------------------------------------------------------------------------------------|
