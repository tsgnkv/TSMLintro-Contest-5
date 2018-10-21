# TSMLintro-Contest-5

В файле stack_pred.py можно найти функцию с одноименным названием, которая реализует стекинг.

Параметры функции stack_pred:
  * estimator - sklearn classifier или regressor
  * X - train data - numpy.array 
  * y - target for train - numpy.array
  * Xt - test data - numpy.array
  * k - number of folds
  * method - 'predict' или 'predict_proba'

По такому принципу реализуется стекинг в функции:
![Реализованный вариант стекинга](https://jeddy92.github.io/images/stack_diagram.png)
