# Audio Sentiment Analysis after a single-channel Multiple Source Separation

The aim of this project is to evaluate the sentiment of a customer throughout a conversation with a call centre agent. Analyzing sentiment of the customer over various parts of the call can help in understanding the transition of customer's emotion.

### Team Members

Arpit Shah, Shivani Firodiya

## Motivation

Call Centers or Support Centers in different companies aggregate huge amount of data everyday. From all the conversations, few conversations are not customer satisfactory. Finding the sentiment of the customer helps in determining whether the customer was satisfied with the service or not. 

## Dataset

* Data for source separation was taken from EXOTEL which consists of 300 audio files. Each of these files contains conversation between the customer and agent on various topics.

* RAVDESS dataset was used for classification purpose, it consists of 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. It can be found [here](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) or [here](https://smartlaboratory.org/ravdess/). Dataset consists of different emotions like -  *neutral*, *calm*, *happy*, *sad*, *angry*, *fearful*, *disgust*, *surprised*.

## Process

We approach this problem in three stages:

**Stage 1** - We perform source separation on the audio conversation, by performing VAD detection on the conversation and dividing the audio conversation into different chunks, on each chunk we apply GMM and a global GMM (UBM) on the whole conversation, using BIC and through spectral clustering we cluster every chunk into different speakers.

**Stage 2** – We then apply sentiment analysis on supervised speech emotion dataset (RAVDESS) using Deep Neural Networks.

**Stage 3** - We used the trained model from Stage 2 to classify the sentiment of the speaker chunks.

## File description

`convert.py` helps convert *.mp3* audio files into *.wav*

`extract.py` is used to extract files from the RAVDESS dataset and store it together on the basis of emotions

`speaker_diarization.py` is used to separate audio chunks of customer and call centre agent. Once the chunks are separated, only chunks containing customer's voice are considered for sentiment analysis.

`sentiment_classification.py` contains complete code on how various audio features that are extracted to train and test the model along with different architectures created to carry out the experiments.

## Miscellaneous Information

Please go through the [poster]() to get an overview of our project and also the different models we used and also the [report]() to learn more about the various papers we referred and the architecture specifications.