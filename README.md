# 2110446-DS-and-DE
code in DS and DE class 

## Table of Contents
1. [1st week](#1st-week)
    1. [key pandas command](#key-pandas-command)
    2. [key datetme transformation](#key-datetme-transformation)
    3. [Assignment 1](#assignment-1)
2. [Reference](#reference)

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

## Reference
- [class github](https://github.com/pvateekul/2110446_DSDE_2023s2)