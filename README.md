# The implementation for machine reading and strategique reasoning

For a given professional article, through machine reading, it is required to be able to locate, analyze, and reason the answers to specific questions in the text. The problem covers six types: factual questions, list-type questions, definition-type questions, opinion-type questions, and text-type questions.
The principle is to let the machine read the above 6 types of questions and generate answers. The matching degree of the answers with the standard answers is evaluated using the ROUGE-L and BLEU indicators. 
For the usage and the results of data, all rights reserved to the organiser.
This implementation is inspired by Google's [QANet](https://openreview.net/pdf?id=B14TlG-RW) and the [blog](https://medium.com/@minsangkim/implementing-question-answering-networks-with-cnns-5ae5f08e312b).

## Dataset
The basic data collection is originally provided by China Electronics Technology Group Corporation No.28 Research Institute, exclusively for participants engaged.

## Requirements

  * Python>=2.7
  * NumPy
  * tqdm
  * TensorFlow>=1.5
  * spacy==2.0.9
  * bottle 

## Usage
To preprocess the data, run

```bash
# preprocess the data
python config.py --mode prepro
```

This procedure of processing is learned from [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net), hyper parameters are stored in config.py. 
To debug/train/test/demo, run

```bash
python config.py --mode debug/train/test/demo
```

To evaluate the trained model with the code provided by the organiser, run
```bash
python test_common.py ~/data/{model_name}.json train/{model_name}/answer/answer.json
```

The default directory for the tensorboard log file is `train/{model_name}/event`
