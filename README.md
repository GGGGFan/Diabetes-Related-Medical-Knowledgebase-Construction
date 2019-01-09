# Diabetes-Related-Medical-Knowledgebase-Construction
Alibaba's AI Challenge: Diabetes-related Medical Knowledgebase Construction

The competition has 2 stages where the topic of the first stage is Named Entity Recognition (NER) from diabetes literature and the topic of the second stage is Relation Extraction (RE) from these literature.

A BiLSTM+CRF model is implemented for the NER task, which achieves an F1 score of 0.717 and ranks 72nd of 1629 teams in Stage 1. For the RE task, a stacking model is built which combines outputs from one BiLSTM+Attention model and one PCNN+Attention model as well as some hand-crafted features. This stacking model finally achieves an F1 score of 0.625 on the leaderboard and 39th in Stage 2.
