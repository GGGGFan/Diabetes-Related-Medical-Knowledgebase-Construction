### Leaderboard
* 39th of 1629, F1 score of 0.625

### Pipeline
* In order to maintain the independency among features, attention-based models are trained with different data.
![stacking](https://user-images.githubusercontent.com/22106895/50928373-1d9f5100-1420-11e9-87c2-d666a0cdf5ed.png)
### Useage
* pre_nn.py: Split literatures into sentences and embed words and positions <br/>
* train_LstmAtt.oy: train BiLSTM+Attention model<br/>
* train_PcnnAtt.py: train PCNN+Attention model<br/>
* mega_train.py: Use the output of attention-based models as well as five hand-crafted features to train a LightGBM model.<br/>
* test_write.py: make predictions and write submission files.<br/>
### Notice
The F1 score can be over 0.72 with 50000 epochs on LightGBM. The 0.625 F1 score is achieved from 5000 epochs.(I forgot a ZERO when setting parameters for the final submission!)
