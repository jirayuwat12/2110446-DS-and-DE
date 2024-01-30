# 2110446-DS-and-DE
code in DS and DE class 

## Table of Contents
1. [1st week](#1st-week)
    1. [key pandas command](#key-pandas-command)
    2. [key datetme transformation](#key-datetme-transformation)
    3. [Assignment 1](#assignment-1)
1. [2nd week](#2nd-week)
    1. [interest command](#interest-command)
1. [Week 3](#week-3)
    1. [AI](#ai)
    1. [Supervise learning (Predictive task)](#supervise-learning-predictive-task)
        1. [Type of problem](#type-of-problem)
    1. [Unsupervise learning](#unsupervise-learning)
    1. [Impurity reduction](#impurity-reduction)
    1. [Tree visualization](#tree-visualization)
    1. [Regularization](#regularization)
    1. [Dicision tree classifier](#dicision-tree-classifier)
    1. [Bagging ( Bootstrap Aggregation )](#bagging--bootstrap-aggregation-)
    1. [Boosting](#boosting)
    1. [Random forest classifier](#random-forest-classifier)
        1. [hyperparameters](#hyperparameters)
    1. [Feature selection from tree (feature importance) with shortcut](#feature-selection-from-tree-feature-importance-with-shortcut)
    1. [Linear regression](#linear-regression)
1. [Week 4](#week-4)
    1. [kNNs](#knns)
        1. [Caution](#caution)
    1. [GridSearchCV](#gridsearchcv)
    1. [RandomizeSearchCV](#randomizesearchcv)
    1. [Save & Load model](#save--load-model)
    1. [Pipelining](#pipelining)
    1. [Neural Network](#neural-network)
        1. [Weight updating](#weight-updating)
    1. [CNN](#cnn)
        1. [example](#example)
1. [Reference](#reference)

## 1st week
### key pandas command
```python
# pandas command

## from 3rd python notebook
map(...)
apply(...)
cut(...)
unique(...)
filter(...)
### key datetme transformation

## from 4th python notebook
file.json(...)
df.merge(...)
```

### Assignment 1
link [here](/assignment_1/5_PandasAssignment.ipynb)

## 2nd week
interest command
```python
import pandas as pd
df = pd.read_csv(...)

# drop row/column
df.dropna(tresh=x) # drop which value not fit treashold
df.drop(columns=[...])

'''
axis 0=row, 1=column
'''

# count by values
df[...].value_counts()

# count Null/None value
df.isnull().sum()

# mapping dict
md = {
    col_name1 : {
        from1 : to1,
        from2 : to2,
        ...
    }
    col_name2 : {...}
}
df.replace(md, inplace=True)

# One-hot encoding
dummied_df = pd.get_dummies(df[col_name], drop_first=...)
#   or use scikit-learn (better)
from sklearn.preprocessing import OneHotEncoder
oh_enc = OneHotEncoder(drop='first')
oh_enc.fit(X)
#   can use with both train and test but use parameter that fitted

# impute missing
from sklearn.impute import SimpleImputer
num_imp = SimpleImputer(missing_value=..., strategy='mean')
num_imp.fit(X)
#   can use with both train and test but use parameter that fitted

# train/test split
#   note that y must be category so if y is numeric -> add grouping column
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=...,
                                                    random_state=42)

# remove outlier
from scipy import stats
#   1. mean +- 3sd (z>3 or z<-3)
z = np.abs(stats.zscore(df))

```

## Week 3

### AI 
has
1. Rule-based AI 
    - Knowledge representation
    - create **Answer** from *data* and *rule*

1. Machine learning
    - create **Rule** from *data* and *answer*
    - `scikit-learn` for traditional

### Supervise learning (Predictive task)
> learn from existed answer

the methodology
1. **Training phase**
    - give data to model

1. **Testing phase (inference)**
    - evaluate perfomance
    - use model in real problem


#### Type of problem
- Classification problem
    - target is **categorical  problem**
    - use **classifier** model
- Regression problem
    - target is **predict numeric**
    - use **regressor** model

### Unsupervise learning
> learn from only data *(no answer)*

### Impurity reduction
1. Entropy
1. Information Gain
    - $\text{Information Gain} = Entropy_{Before} - Entropy_{After}$
1. Gini impurity
    - $Gini = 1-\sum_{i}^{n}(P_i)^2$
        - $P_i$ : prob. of class i in data-set
    - $\text{Gini reduction} = Gini_{Before} - Gini_{After}$

### Tree visualization
```python
from sklearn.tree import plot_tree

plot_tree(model)
```

### Regularization
- balance between **performance** and **complexity**

### Dicision tree classifier
- make decision based on node criteria
- each leaf node represents area and % of confidense is from ratio between class

essentials
- split search *- compare impurity between before and after split* and select best purity from the split
- after get splitted area -> recursive on each area.

hyperparameters
- `max_depth` : maximum depth of the model
    - the more `max_dept`, the more **overfitting**
- `min_leaf_size` : minimum datapoints in each leaf(area)
    - the less `min_leaf_size`, the more **overfitting**

Adventage
- the model is **decribable**
- able to tell **feature importance**
    - summation of $\nabla\text{goodness}$
    - use for **variable selection**
    - `model.feature_importances_`

Be caution
1. Instability
    - very sentitive to datapoints. model change with a little noise

Pruning
- $ R_\alpha (T) = R(T) + \alpha|T|$
- use $\alpha$ to regularization the three
- the more $\alpha$, the smaller tree
- $\alpha$ is `ccp_alphas` in `sklearn.tree` but default is **0**

![importance parameters](./assets/tree_importance_parameters.png)

### Bagging ( Bootstrap Aggregation )
> random with replacement

1. random subset 
    - each subset can be overlapping
1. use subset to train model ( get more model )
1. use each model to help predict together

### Boosting
> onvert **waek learner** to **stronger one**

- in each step, we boost freq. the wrong case of the previous tree.
- do any step until we accept the perfomance

e.g. `AdaBoosting`, `XGBoost`, etc.

### Random forest classifier
> random without replacement

1. random subset
    - all subset must not overlapped both data and features
1. do like normal tree

hyperparameters
1. `max_sample` 
1. `max_featurea`
1. `n_estimators`

#### hyperparameters
1. #Tree
1. #Columns (features)
1. #Rows (example)

### Feature selection from tree (feature importance) with shortcut
```python
from sklearn.feature_selection import SelectFromModel

model = ...
selector = SelectFromModel(model)
selector = select.fit(X, y)
selector.get_support()
```
### Linear regression
assumption
1. **linear relationship** between feature and target
1. error are **independent** from each other
1. target distribution is normal **(no outlier)**
    - error are normally distributed
    - error have constant variance

Regularization
- Idea : $\text{Loss} = \text{Error} + \lambda\text{Complexity}$
- L1 : Lasso
    - **absolutely**
    - $ \text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y_i})^2 + \lambda\sum_{j=1}^{p}|\beta_j|$
- L2 : Ridge
    - **square**
    - $ \text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y_i})^2 + \lambda\sum_{j=1}^{p}\beta_j^2$

Sklean example
    
```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

model = Lasso(alpha=...)

model = Ridge(alpha=...)

model = ElasticNet(alpha=..., l1_ratio=...) # l1_ratio = 0 -> Ridge, l1_ratio = 1 -> Lasso
```

Basic solution
- if not linear -> use **Neural Network**
- if not normal -> take log to make it ***more*** normal

## Week 4

### kNNs
- `k` : number of nearest neighbors -> to make **vote/average/miximum prob, etc.**
- `distance_fn` : to measure distance

#### Caution
1. must be **numeric** value
1. must **normalize** data on each axis

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

grid_search = GridSearchCV(estimator=...,
                           param_grid=...,
                           scoring=...,
                           cv=StratifiedKFold(n_splits=5))
grid_Ssarch.fit(...)

model = grid_search.best_estimator_

model.predict(...)
```

### RandomizeSearchCV

- as same as `GridSearchCV` but randomize approach
- `n_iter` : to tell the most iteration to random select.

### Save & Load model

```python
import pickle

# to save model
pickle.dump(model, open('model.pkl', 'wb'))

# to load model
loaded_model = pickle.load(open('model.pkl', 'rb'))

loaded_model.predict(...)
```

### Pipelining

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

num_pl = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')),
                         ('scale', StandardScaler())])

cat_pl = Pipeline(steps=[('imput', SimpleImputer(strategy='most_freq')),
                         ('scale', OneHotEncoder())])

from sklearn.compose import ColumnTransformer

col_transf = ColumnTransformer(transformers=[('num_pl', num_pl, num_cols),
                                             ('cat_pl', cat_pl, cat_cols),
                                             n_jobs=-1])

model_pl = Pipeline(steps=[('col_trans', col_trains),
                            'model', model])

# display pipeline
display(model_pl)

# to get parameters name
model_pl.get_params()
```

### Neural Network
- `hidden_units`, `hidden_layers`, `lr`, `decay`, etc.
```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

# solver is equal to Optimizer
'''
for example 2 layer that has 100, 200 nodes
si hidden_layer_sizes = (100, 200)
'''
model = MLPClassifier(hidden_layer_sizes=...,
                      actication='relu',
                      solver=...)
```

#### Weight updating
1. Stochastic Gradient Descent (SGD)
    - update weight for every single iteration(each data point)
    - **note** not tolerate to outlier
1. Batch Gradient Descent 
    - train all the data and **average** gradients to calculate back prop.
1. Mini-batch Gradient Descent
    - group data into smaller batch and **average**  bra bra bra
    - **note** use less memory, tolerate to outlier.
    - aka. `batch_size`

### CNN
- kernal
    - `filter size` : size of kernel
    - `filters` : number od kernel
    - `stride` : number of pixel that kernel moved
    - `padding` : number of pixel extended
**process** convolution layer (feature extraction) -> NN -> outcomes

#### example 

1. VGGNet
    - feature extraction layer : `Conv2D(3x3)` + `Conv2D(3x3)` + `MaxPool`
1. Inception V1/ GoogLeNet
    - variant kernel size
    - **Deeper and Wider**
1. Inception V2, V3
    - more speed
    - **Factorize** metrix e.g. 5x5 kernal represented by 2 of 3x3 kernel, etc.
    - use **Batch norm** to reduce *Gradient vanishing*
1. ResNet 
    - make **Short skip connection** then each group call *residual block*, to reduce *Gradient vanishing*
1. Inception-ResNet
    - make Inception go deeper.
1. EfficientNet
    - deeper, wider, resolution (variant size of kernel) -> **compound scaling** 
1. EfficientNet V2
    - smaller and faster 6x
    - size : S, M, L, XL


## Reference
- [class github](https://github.com/pvateekul/2110446_DSDE_2023s2)