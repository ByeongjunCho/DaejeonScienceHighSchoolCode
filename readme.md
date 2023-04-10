# 대전과학고 심화자율연구를 위한 기초코드

## 2021

1. modeling_bert_method1.py

   * Word embedding과 Position embedding이 분린되어 계산되는 Transformer 기반 모델
   * 해당 모델 사용 시 Transformer 파일에서 custom 한 코드이므로 `transformers_\models\bert` 에 넣어줘야 정상적으로 동작함
   * Method 1이 구현되어 있음 - attention 계산 시 word/position embedding을 각각 받아서 계산

2. original_BERT_training.py

   * wikitext-103 data를 BERT model에 pretraining 하는 toy code

   * Pretrained Tokenizer + BERT를 사용하여 pretraining

   * 시간 절약을 위해, NO NSP(Next Sentence Prediction) 사용 - 성능에 미치는 영향 적음

   * 6월 3일 발표자료 참고

     ![img](https://blog.kakaocdn.net/dn/T33yE/btrvaJbjPFk/C4hJp2a4SGMD1KxlteR9q0/img.png)

## 2022

[lstm_example.ipynb](https://github.com/ByeongjunCho/DaejeonScienceHighSchoolCode/blob/main/2022/lstm_example.ipynb) : LSTM 주식 예측 모델

[seq2seq_stock-5.ipynb](https://github.com/ByeongjunCho/DaejeonScienceHighSchoolCode/blob/main/2022/seq2seq_stock-5.ipynb) : Seq2seq model 주식 예측 모델 예제

[Transformer_Stock_NoneStationality.ipynb](https://github.com/ByeongjunCho/DaejeonScienceHighSchoolCode/blob/main/2022/Transformer_Stock_NoneStationality.ipynb) : 주식에서 사용되는 다양한 indicator를 이용한 예제. Transformer encoder + Linear layer 로 구성. Pretrained transformer 는 text로 pretrained 되어 있으므로 시계열 모델에 pretrained model 사용이 적합하지 않음

