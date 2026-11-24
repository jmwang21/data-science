import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone


def house_price_workflow(train, val, test):
    for frame in (train, val, test):
        assert frame.columns.is_unique
        assert frame['id'].notna().all() and frame['id'].is_unique
    assert 'price' not in test.columns
    assert train.price.notna().all() and val.price.notna().all()
    features = train.columns.drop(['id', 'price']).tolist()
    assert set(val.columns) == set(train.columns)
    assert set(test.columns) == set(train.columns) - {'price'}
    X, y = train[features], train.price
    Xv, yv = val[features], val.price
    Xt = test[features]
    num = X.select_dtypes(include='number').columns.tolist()
    cat = [c for c in features if c not in num]
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), num),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ])
    specs = {
        'baseline': (DummyRegressor(strategy='median'), {}),
        'ridge': (Ridge(), {'model__alpha': [.1, 1, 10]}),
        'rf': (RandomForestRegressor(n_estimators=100, random_state=42),
               {'model__min_samples_leaf': [1, 5], 'model__max_depth': [8, None]}),
        'gb': (GradientBoostingRegressor(loss='absolute_error', random_state=42),
               {'model__n_estimators': [100, 200], 'model__learning_rate': [.05, .1]})
    }
    cv = KFold(5, shuffle=True, random_state=42)
    fitted, scores = {}, {}
    for name, (est, params) in specs.items():
        search = GridSearchCV(Pipeline([('pre', clone(pre)), ('model', est)]), params,
                              cv=cv, scoring='neg_mean_absolute_error', n_jobs=1)
        search.fit(X, y)
        fitted[name] = search
        scores[name] = mean_absolute_error(yv, search.predict(Xv))
    best = min(scores, key=scores.get)
    final = clone(fitted[best].best_estimator_).fit(
        pd.concat([X, Xv], ignore_index=True), pd.concat([y, yv], ignore_index=True))
    submission = pd.DataFrame({'id': test.id.to_numpy(), 'price': final.predict(Xt)})
    assert len(submission) == len(test)
    assert submission.id.tolist() == test.id.tolist()
    assert np.isfinite(submission.price).all()
    return submission, scores, best, fitted[best].best_params_


def synthetic_data():
    rng = np.random.default_rng(42)
    n = 200
    area = rng.integers(40, 200, n).astype(float)
    age = rng.integers(0, 40, n)
    district = rng.choice(['central', 'outer'], n)
    price = 50_000 + 3200*area - 900*age + 80_000*(district == 'central') + rng.normal(0, 20_000, n)
    data = pd.DataFrame({'id': np.arange(n), 'area': area, 'age': age, 'district': district, 'price': price})
    data.loc[[3, 90, 180], 'area'] = np.nan
    train = data.iloc[:140].copy()
    val = data.iloc[140:170].copy()
    test = data.iloc[170:].drop(columns='price').copy()
    test.loc[170, 'district'] = 'unseen'
    return train, val, test

if __name__ == '__main__':
    train, val, test = synthetic_data()
    submission, scores, best, params = house_price_workflow(train, val, test)
    assert len(submission) == 30 and list(submission.columns) == ['id', 'price']
    assert scores[best] == min(scores.values())
    print('validation MAE:', {k: round(v,2) for k,v in scores.items()})
    print('best:', best, params)
    print('submission:', submission.shape, 'finite:', np.isfinite(submission.price).all())
