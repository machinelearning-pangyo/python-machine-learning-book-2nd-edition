# 그리드 서치를 이용한 하이퍼파라미터 튜닝

하이퍼파라미터를 최적화하면 모델 성능을 향상시키는데 큰 도움이 됩니다.  
그리드 서치는 리스트로 지정된 여러 하이퍼파라미터 값을 받아 모든 조합에 대해 모델 성능을 평가하여 최적의 하이퍼파라미터 조합을 찾습니다.

```
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
// 0.9846859903381642
print(gs.best_params_)
// {'svc__C': 100.0, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'}
```

> 그리드 서치는 최적의 하이퍼파라미터를 찾는데 유용한 도구이지만, 모든 조합에 대해 평가를 진행하기 때문에 계산 비용이 매우 비쌉니다.  
> 사이킷런은 RandomizedSearchCV 클래스를 사용하여 제한된 횟수 안에서 랜덤한 매개변수 조합을 뽑아 학습하는 랜덤 서치 방법을 제공합니다.

### 중첩 교차 검증을 사용한 알고리즘 선택

여러 종류의 머신러닝 알고리즘을 비교할때는 `중첩 교차 검증(nested cross-validataion)`이 권장되며, `그리드 서치`와 `k-겹 교차 검증`을 함께 사용하면 모델의 성능을 세부 튜닝하기에 좋습니다.

![nested cross-validataion](http://cfile27.uf.tistory.com/image/990466355E24F98231530A)

바깥쪽 k-겹 교차 검증 루프가 데이터를 훈련 폴드와 테스트 폴드로 나누고,  
안쪽 루프가 훈련 폴드에서 k-겹 교차 검증을 수행하며 모델을 선택합니다.  
모델이 선택되면 테스트 폴드를 사용하여 모델 성능을 평가합니다.

#### 위 SVM 모델 이용시

```
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))
// CV 정확도: 0.974 +/- 0.015
```

#### 단일 결정 트리 이용시

```
from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy', cv=5)
print('CV 정확도: %.3f +/- %.3f' % (np.mean(scores), 
                                      np.std(scores)))
// CV 정확도: 0.934 +/- 0.016
```
