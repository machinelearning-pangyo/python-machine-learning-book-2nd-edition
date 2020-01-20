# 머신러닝 모델의 성능 평가 지표

### 오차 행렬(Confusion Matrix)

-   진짜 양성 (TN : True Positive)
-   진짜 음성 (TN : True Negative)
-   거짓 양성 (FP : False Positive)
-   거짓 음성 (FN : False Negative)

상세 예제 : [구글 머신러닝 단기 집중과정](https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative?hl=ko)

오차 행렬은 사이킷런에서 제공하는 `confusion_matrix`함수를 이용하여 편리하게 사용할수 있습니다.

```
from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
// [[71  1]
//  [ 2 40]]
```

예측 오차와 정확도는 얼마나 많은 샘플을 잘못 분류 했는지 알려줍니다.

#### 예측 오차 (ERR)

( FP + FN ) / (FP + FN + TP + TN)

#### 예측 정확도 (ACC)

( TP + TN ) / (FP + FN + TP + TN)  
\= 1 - ERR

TPR과 FPR은 오차 행렬에서 행(실제 클래스)끼리 계산하기 때문에 클래스 비율에 영향을 받지 않아 유용한 성능 지표입니다.

#### 진짜 양성 비율 (TPR)

TP / P = TP / (FN + TP)

#### 거짓 양성 비율 (FPR)

FP / N = FP / (FP + TN)

정확도와 재현율 성능 지표는 진짜 양성과 진짜 음성 샘플의 비율과 관련이 있습니다.

#### 정확도 (PRE)

TP / (TP + FP)

#### 재현율 (REC)

REC = TPR

실전에서는 PRE와 REC를 조합한 `F1-점수`를 자주 사용합니다.

#### F1-점수

2 \* (PRE \* REC) / (PRE + REC)

sklearn은 metrics 모듈을 이용하여 위 지표를 제공합니다.

```
from sklearn.metrics import precision_score, recall_score, f1_score

print('정밀도: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('재현율: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
// 정밀도: 0.976
// 재현율: 0.952
// F1: 0.964
```

### ROC 곡선

![ROC 곡선](http://cfile7.uf.tistory.com/image/9938F6425E252494332B4F)  
ROC 곡선(Receiver Operation Characteristic: 수신자 조작 특성 곡선)은 TPF, FPR 점수를 기반으로 모든 분류 임계값에서 분류 모델의 성능을 보여주는 그래프입니다.  
완벽한 분류기의 그래프는 TPR이 1이고 FPR이 0인 왼쪽 위 구석에 위치합니다.

### AUC: ROC 곡선 아래 영역

![AUC](http://cfile4.uf.tistory.com/image/99D1413E5E25716338B5C0)  
AUC는 곡선 아래 영역을 나타냅니다.  
AUC가 넓을수록, 안정된 예측을 할 수 있는 모델입니다.