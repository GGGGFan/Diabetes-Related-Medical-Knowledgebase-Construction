### Useage
* pre_nn.py: Split literatures into sentences and embed words and positions <br/>
* train_LstmAtt.oy: train BiLSTM+Attention model<br/>
* train_PcnnAtt.py: train PCNN+Attention model<br/>
* mega_train.py: Use the output of attention-based models as well as five hand-crafted features to train a LightGBM model.<br/>
* test_write.py: make predictions and write submission files.<br/>
### Notice
* In order to maintain the independency among features, attention-based models are trained with different data.
