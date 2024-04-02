# UNSURE: understanding user confidence in spoken queries for conversational search systems
This repository provides the code for the paper titled **[UNSURE: understanding user confidence in spoken queries for conversational search systems]()**, making the integration of our code contributions into other projects more accessible.

## Quick Links
- [Overview](#abstract)
- [Dataset: UNSURE](#dataset-unsure)
  - [Audio generation](#audio-generation)
  - [Annotated scores](#annotated-scores)
- [Training](#training)
- [Citation](#citation)


## Abstract
The confidence level in users’ speech has long been recognised as an important signal in traditional dialogue systems. In this work, we highlight the importance of user confidence detection in queries in conversational search systems (CSSs). Accurately estimating a user’s confidence level in CSSs is important because it enables the CSSs to inferthe degree of competency of a user on the queried topic and subsequently tailor its responses appropriately. This is especially important in CSSs since their responses need to be concise and precise. However, few prior works have evaluated user confidence in CSSs due to a lack of available datasets. We present a novel speech-based dataset named UNderstanding Spoken qUeRiEs (UNSURE), which contains confidence grading annotations of user queries in natural language conversations. Based on this dataset, we propose a multimodal approach to infer users’ confidence in spoken queries as a baseline model. Preliminary experimental results demonstrate that our proposed fusion model is capable of achieving near human-level performance.

![model_new](https://github.com/YoujingYu99/confidence_css/assets/67215422/62e16890-6a01-452f-9383-be339a221e5c)


## Dataset: UNSURE
UNSURE is a dataset prepared from the [Spotify Podcast Dataset](https://podcastsdataset.byspotify.com/). Each data point in UNSURE is an audio excerpt and three confidence scores, annotated by Amazon MTurk workers. Here we will briefly describe how to re-produce the dataset. 

### Audio generation
If you have access to the dataset:
Please specify the `home_dir`, `audio_dir` and `extracted_dir` in the following script before running it
```sh
python data_preparation/audio_excerpt_pointer.py
```

The audio excerpts will be generated. The `confidence_dataframe_total_remove.csv` file in the folder `data/confidence_dataframes/` contains the following columns:
*  `start_time`: start time of the sentence in the audio file.
*  `end_time`: end time of the sentence in the audio file.
*  `sent_end_time`: actual end time of the sentence/start of next sentence.
*  `sentence`: text transcript.
*  `filename`: filename/location of the audio file.
*  `category`: category of the audio file.
*  `inter_freq`: frequency of interjecting sounds.
This `confidence_dataframe_total_remove.csv` file is used for the generation of the audio excerpts.


### Annotated scores
In the folder `data/label_results` there are four files: `Cleaned_Results_Removed.csv`, `Benchmark_Samples_Removed.csv`, `Human_Labels_Removed.csv` and `Cleaned_Results_Test_Removed.csv`. We shall briefly explain what each file contains. For more details please refer to the paper. 

* `Cleaned_Results_Removed.csv` constains the full labels corresponding to the audio files. It contains the following columns:
  *  `audio_url`: folder + name of the audio. It follows the same convention as the audio generated from the previous step.
  *  `score1/2/3`: confidence scores given by three MTurk workers.
  *  `average`: average of the three scores.
    
* `Benchmark_Samples_Removed.csv` contains the samples used for benchmarking. It contains the following columns:
  *  `audio_url`: folder + name of the audio. It follows the same convention as the audio generated from the previous step.
  *  `Answer.questionVerify`: 1 if the author thinks the speaker is asking a question.
  *  `Answer.speakerNumberVerify`: 1 if the author thinks there are multiple speakers in the audio
    
* `Human_Labels_Removed.csv` contains the scores labelled by the expert (the author) on the test dataset. 
  *  `audio_url`: folder + name of the audio. It follows the same convention as the audio generated from the previous step.
  *  `score4`: confidence scores given by the author.
    
* `Cleaned_Results_Test_Removed.csv` contains the same columns as `Cleaned_Results_Removed.csv`. The only difference is that it is the test dataset and the audio clips used are the same as those in `Human_Labels_Removed.csv`.

## Training
The train and test dataset can be generated by calling the `split_to_train_val` function in  `network_model/model_utils.py`. Note that we should only shuffle and sample training and validation datasets. We should use `Cleaned_Results_Test_Removed.csv` as the dataframe specifying the test dataset as it contains the same audios as used in `Human_Labels_Removed.csv` (expert validation set). 

To train the audio-text fusion model,  specify `home_dir` in `network_model/audio_text.py` then run:
```sh
python network_model/audio_text.py
```

To train the ablation model, specify `home_dir` and which ablation (audio or text) test by `ablate_text` to True or False in `network_model/audio_text_ablation.py` then run 
```sh
python network_model/audio_text_ablation.py
```

## Citation
```
@inproceedings{yu2024unsure,
title={UNSURE: understanding user confidence in spoken queries for conversational search systems},
author={Youjing Yu and Zhengxiang Shi and Aldo Lipani},
booktitle={25th International Conference on Engineering Applications of Neural Networks (EANN/EAAAI)},
year={2024},
url={https://github.com/YoujingYu99/confidence_css}
}
```