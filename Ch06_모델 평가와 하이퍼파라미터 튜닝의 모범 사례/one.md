# sklearn Pipeline 파이프라인

사이킷런의 Pipeline 클래스는 연속된 변환을 순차적으로 처리할 수 있는 기능을 제공하는 유용한 래퍼(Wrapper) 도구입니다.

```
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(solver='liblinear', random_state=1))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('테스트 정확도: %.3f' % pipe_lr.score(X_test, y_test))
```

`make_pipeline` 함수는  
사이킷런 변환기(입력에 대해 fit 메서드와 transform 메서드를 지원하는 객체)와  
그 뒤에 fit 메서드와 predict 메서드를 구현한 사이킷런 추정기를 연결할수 있습니다.

![sklearn pipeline](http://cfile10.uf.tistory.com/image/99EF85365E24E3A41A41DE)

파이프라인의 `fit()` 메서드를 호출하면 모든 변환기의 `fit_transform()` 메서드를 순차적으로 호출하면서 각 단계의 output을 다음 단계의 input으로 전달합니다.  
마지막 단계에서는 `fit()` 메서드만 호출하게 됩니다.

파이프라인의 `predict()` 메서드를 호출하면 매개변수로 전달된 데이터가 각 단계의 `transform()`메서드를 통과하게 됩니다.  
마지막 단계에서는 추정기 객체가 변환된 데이터에 대한 예측 값을 반환합니다.
