# ImagiNet: A Balanced High-Resolution Dataset for Benchmarking of Synthetic Image Detectors
![](media/dataset_preview.png)
**Abstract:**
<p align="justify">Generative models, such as Diffusion Models (DMs), Variational    Auto-Encoders (VAEs), and Generative Adversarial Networks (GANs),    produce images with a level of authenticity that makes them nearly    indistinguishable from real photos and artwork. While this capability    is beneficial for many industries, the difficulty of identifying    synthetic images leaves online media platforms vulnerable to    impersonation and misinformation attempts. To support the development    of defensive methods, we introduce ImagiNet, a high-resolution and    balanced dataset for synthetic image detection. It contains 200K    examples, spanning four content categories: photos, paintings, faces,    and uncategorized. Synthetic images are produced with open-source and    proprietary generators, whereas real counterparts of the same content    type are collected from public datasets. The structure of ImagiNet    allows for a two-track evaluation system: i) classification as real    or synthetic and ii) identification of the generative model. To    establish a baseline, we train a ResNet-50 model using a    self-supervised contrastive (SelfCon) objective for each track. The    model demonstrates strong performance across previously available    benchmarks, achieving an AUC of up to 0.97 and balanced accuracy    ranging between 80% and 93%, even under social network conditions    involving compression and resizing. </p>

## Instructions for downloading
For now we have provided only an automatic download script for **FFHQ** datasets since we used custom filenames. Before downloading, you should write the path to your credentials for Google Drive API in `download_scripts/ffhq.py` script and install the packages from `requirements.txt`. Other datasets should be downloaded the datasets from their original sources and placed in one directory. This is the root directory of the dataset. After that, you can invoke `dataset_operations/delete_not_needed.py` with an argument to the path of the root directory where all datasets are extracted. The script will remove the images that are not included in our dataset.
| :information_source: You should place all extracted datasets in one folder. |
|-----------------------------------------------------------------------------|

The extracted content should be put in the following folders (<sup>*</sup> signifies datasets that do not require additional folders to be created):
 - **ImageNet:** imagenet (ILSVRC folder placed here)
 - **COCO<sup>*</sup>:** train2017, val2017
 - **LSUN/ProGAN<sup>*</sup>:** progan_train, progan_val
 - **WikiArt<sup>*</sup>:** wikiart
 - **LSUN/ProGAN<sup>*</sup>:** progan_train, progan_val
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

## Results
In the following tables ACC is the balanced acccuracy and AUC is the Area Under the Receiver Operating Characteristics. Best ACC is signified with **bold** and best AUC with <ins>underline</ins>.

### Synthetic Image Detection

#### Results group of models 
|  ACC / AUC |     Grag2021    |    Corvi2022    |      Wu2023     |       Ours      |
|:----------:|:---------------:|:---------------:|:---------------:|:---------------:|
| GAN        | 0.6889 / 0.8403 | 0.6822 / 0.8033 | 0.6508 / 0.6971 | **0.8242** / <ins>0.8871</ins> |
| SD         | 0.5140 / 0.5217 | 0.6112 / 0.6851 | 0.6367 / 0.6718 | **0.9145** / <ins>0.9724</ins> |
| Midjourney | 0.4958 / 0.5022 | 0.5826 / 0.6092 | 0.5326 / 0.5289 | **0.8844** / <ins>0.9448</ins> |
| DALL·E 3   | 0.4128 / 0.3905 | 0.5180 / 0.5270 | 0.5368 / 0.5482 | **0.9120** / <ins>0.9709</ins> |
| **Mean**       | 0.5279 / 0.5637 | 0.5985 / 0.6562 | 0.5892 / 0.6116 | **0.8838** / <ins>0.9438</ins> |

#### Results specific models 
|    ACC / AUC   |     Grag2021    |    Corvi2022    |      Wu2023     |       Ours      |
|:--------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| ProGAN         | 0.7270 / 0.9319 | 0.7263 / 0.9262 | 0.8147 / 0.9520 | **0.9540** / <ins>0.9993</ins> |
| StyleGAN-XL    | 0.6747 / 0.8048 | 0.6456 / 0.7584 | 0.5851 / 0.6076 | **0.8905** / <ins>0.9563</ins> |
| StyleGAN3      | 0.6924 / <ins>0.8412</ins> | **0.7031** / 0.7795 | 0.6682 / 0.6877 | 0.6329 / 0.6973 |
| SD v2.1        | 0.5375 / 0.5579 | 0.6196 / 0.6908 | 0.6261 / 0.6555 | **0.9001** / <ins>0.96203</ins> |
| SDXL v1.0      | 0.5653 / 0.6008 | 0.6333 / 0.7295 | 0.6160 / 0.6449 | **0.9418** / <ins>0.99133</ins> |
| Animagine XL   | 0.2862 / 0.1711 | 0.5231 / 0.5346 | 0.7289 / 0.7836 | **0.8831** / <ins>0.95403</ins> |
| Midjourney     | 0.4886 / 0.4946 | 0.5690 / 0.6021 | 0.5346 / 0.5296 | **0.8818** / <ins>0.94363</ins> |
| DALL·E 3       | 0.4094 / 0.3798 | 0.5152 / 0.5253 | 0.5448 / 0.5642 | **0.9094** / <ins>0.96983</ins> |
| **Mean**       | 0.5476 / 0.5978 | 0.6169 / 0.6933 | 0.6398 / 0.6781 | **0.8742** / <ins>0.93423</ins> |


#### Results content types
|   ACC / AUC   |     Grag2021    |    Corvi2022    |      Wu2023     |       Ours      |
|:-------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Photos        | 0.6676 / 0.7593 | 0.7203 / 0.8247 | 0.6933 / 0.7533 | **0.8242** / <ins>0.8871</ins> |
| Paintings     | 0.5637 / 0.6158 | 0.6074 / 0.6853 | 0.5030 / 0.4614 | **0.9145** / <ins>0.8402</ins> |
| Faces         | 0.5781 / 0.6675 | 0.5316 / 0.6913 | 0.6630 / 0.7257 | **0.8844** / <ins>0.9448</ins> |
| Uncategorised | 0.4299 / 0.4110 | 0.6010 / 0.6431 | 0.5847 / 0.6091 | **0.9120** / <ins>0.9709</ins> |
| **Mean**      | 0.5598 / 0.6134 | 0.6151 / 0.7111 | 0.6110 / 0.6374 | **0.8838** / <ins>0.9438</ins> |
### Model Identification
#### Results of our Synthetic Generator Identificator (in terms of ACC)
|  GAN  | SD |    Midjourney    |      DALL·E 3     |      Mean     |
|:----:|:----:|:----:|:----:|:----:|
| 0.9998 | 0.9978 | 0.9964 | 0.9872 | 0.9953 |
 
AUC = 0.9999

### Inference Time
Inference testing is conducuted in identical conditions - 448x448 input image. All tests are conducted on a single 4090 GPU. The best inference time is signified with **bold**.
|  Detector | Inference Time (ms) ↓ |
|:---------:|:-------------------:|
| Grag2021  |        24.23        |
| Corvi2022 |        48.75        |
| Wu2023    |        15.92        |
| Ours      |       **1.84**      |


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
