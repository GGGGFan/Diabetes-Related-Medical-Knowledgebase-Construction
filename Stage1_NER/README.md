### CODES ARE ADUJSTED FOR ALIBABA'S REVIEW
### Model:
Random Embedding on Character level + BiLSTM + CRF
### Requirement:<br/>
python 3.6<br/>
Tensorflow 1.8.0<br/>
Keras 2.2.4<br/>
Keras_contrib 0.0.2<br/>
(Keras is used in this stage for easier CRF implementation)
### Usage:<br/>
predict.py: Load trained model and predict<br/>
main.py: Train from raw data and predict<br/>

Note: Due to the use of GPU, results may differ from submission if model is re-trained from raw data even though random seed is set.
