from scipy.stats import randint,uniform

# LightGBM hyperparameter search space for RandomizedSearchCV
LIGHTGM_PARAMS={
    'n_estimators':randint(100,500),
    'max_depth' : randint(5,50),
    'learning_rate': uniform(0.01,0.2),
    'num_leaves': randint(20,100),
    'boosting_type' : ['gbdt' , 'dart' , 'goss'],
    'min_child_samples': randint(10,100),
    'subsample': uniform(0.6,0.4),
    'colsample_bytree': uniform(0.6,0.4)
}

# RandomizedSearchCV settings
RANDOM_SEARCH_PARAMS = {
    'n_iter' : 10,
    'cv' : 3,
    'n_jobs':-1,
    'verbose' :2,
    'random_state' : 42,
    'scoring' : 'f1'
}