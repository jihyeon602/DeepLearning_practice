**옵티마이저 API**

다양한 옵티마이저의 장단점에 대해서는 [Hands-on Machine Learning 3판](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)의 11장에 정리되어 있다.

| 옵티마이저               | 문자열    |
| :----------------------- | :-------- |
| keras.optimizers.SGD     | “SGD”     |
| keras.optimizers.RMSprop | “rmsprop” |
| keras.optimizers.Adam    | “adam”    |
| keras.optimizers.AdamW   | “adamw”   |
| keras.optimizers.Adagrad | “adagrad” |
| keras.optimizers.Nadam   | “nadam”   |

**손실 함수 API**

일반적으로 모델의 종류에 따라 손실 함수를 선택한다.

| 손실 함수                                  | 문자열                            | 용도                                 |
| :----------------------------------------- | :-------------------------------- | :----------------------------------- |
| keras.losses.CategoricalCrossentropy       | “categorical_crossentropy”        | 다중 클래스 분류. 원-핫 형식 타깃    |
| keras.losses.SparseCategoricalCrossentropy | “sparse_categorical_crossentropy” | 다중 클래스 분류. 정수 타깃          |
| keras.losses.BinaryCrossentropy            | “binary_crossentropy”             | 이진 분류                            |
| keras.losses.MeanSquaredError              | “mean_squared_error”              | 회귀                                 |
| keras.losses.MeanAbsoluteError             | “mean_absolute_error”             | 회귀                                 |
| keras.losses.CosineSimilarity              | “cosine_similarity”               | 문장 번역, 물건 추천, 이미지 분류 등 |

**평가지표 API**

일반적으로 모델 종류와 목적에 따라 평가 지표를 선택한다.

| 평가지표                                | 문자열                        | 용도                                         |
| :-------------------------------------- | :---------------------------- | :------------------------------------------- |
| keras.metrics.CategoricalAccuracy       | “categorical_accuracy”        | 다중클래스 분류 정확도 측정. 원-핫 형식 타깃 |
| keras.metrics.SparseCategoricalAccuracy | “sparse_categorical_accuracy” | 다중클래스 분류 정확도 측정. 정수 타깃       |
| keras.metrics.BinaryAccuracy            | “binary_accuracy”             | 이진 분류 정확도 측정                        |
| keras.metrics.AUC                       | 없음                          | 다중 클래스 분류 AUC 측정                    |
| keras.metrics.Precision                 | 없음                          | 다중 클래스 분류 정밀도 측정                 |
| keras.metrics.Recall                    | 없음                          | 다중 클래스 분류 재현율 측정                 |