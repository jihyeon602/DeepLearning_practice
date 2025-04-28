### 7.4.2. 규제: 드롭아웃

모델이 훈련셋에 너무 익숙해지지 않도록 방해하는 것을 규제라 부른다. 모델 규제를 위해 **드롭아웃**drop out 기법이 많이 적용된다. 드롭아웃은 무작위로 선택된 일정한 비율의 유닛을 끄는 것을 의미하며, 해당 유닛에 저장된 값을 0으로 처리한다.

아래 그림은 50%의 드롭아웃을 층의 출력값에 적용하는 과정을 보여준다. 출력값의 50%를 0으로 처리한 다음에 출력 텐서를 0.5로 나눈다. 즉 2를 곱해준다. 이유는 드롭아웃 기능은 훈련에만 적용하며 실전에서는 사용되지 않기에 출력값의 최종 크기를 훈련일 때와 아닐 때 비슷하게 만들어주기 위해서이다.

![img](https://drek4537l1klr.cloudfront.net/chollet2/Figures/05-20.png)

<그림 출처: [Deep Learning with Python(2판)](https://www.manning.com/books/deep-learning-with-python-second-edition)>

케라스에서는 드롭아웃 적용을 위해 적절한 드롭아웃 비율을 답은 드롭아웃 층 `Dropout`을 활용한다. 아래 그래프는 IMDB 영화 후기 분류 모델 대상으로 50%의 드롭아웃을 적용하여 훈련하면 검증셋에 대한 손실값이 보다 천천히 증가함을 보여준다. 즉, 모델의 과대적합이 보다 늦게, 보다 약하게 발생한다.

![img](https://drek4537l1klr.cloudfront.net/chollet2/v-7/Figures/original_model_vs_dropout_regularized_model_imdb.png)

<그림 출처: [Deep Learning with Python(2판)](https://www.manning.com/books/deep-learning-with-python-second-edition)>
