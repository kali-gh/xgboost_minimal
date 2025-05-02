import json
import logging
import pandas as pd
import urllib.request
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.info("initializing")

url = 'https://archive.ics.uci.edu/static/public/73/mushroom.zip'
input_data_path = 'input_data'
input_data_file_name = 'mushroom'
filename_data = 'agaricus-lepiota.data'
filename_data_names = 'agaricus-lepiota.names'
filename_column_names = 'columns.json'
LOAD_DATA = False

n_estimators = 5
learning_rate = 0.001
max_depth = 2
random_state = 42

Path(input_data_path).mkdir(exist_ok=True, parents=True)

def load_data() -> pd.DataFrame:
    """
    Loads the data from the url provided
    :return: dataframe with the data
    """

    if LOAD_DATA:
        filename_zip = f'{input_data_path}/{input_data_file_name}.zip'
        logger.info(f"Loading data from {url}, saving to {filename_zip}")
        urllib.request.urlretrieve(url, filename_zip)

        with zipfile.ZipFile(filename_zip, 'r') as zip_ref:
            zip_ref.extractall(input_data_path)
    else:
        logger.info(f"Loading data from {input_data_path}")

    df = pd.read_csv(f'{input_data_path}/{filename_data}', header=None)

    # copied from the .names file provided in the url at the top for convenience..
    with open(f"{input_data_path}/{filename_column_names}", 'r') as f:
        columns = json.load(f)

    df.columns = columns

    logger.info("Loaded data: ")
    logger.info(df.head())

    return df

def build_featurized_df(
        df : pd.DataFrame) -> pd.DataFrame:
    """
    Builds a featurized dataframe from the given dataframe
    :param df: dataframe to handle
    :return: featurized dataframe
    """
    categorical_features = df.columns.copy()

    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=True
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),

    ])

    df_array = pipeline.fit_transform(df)
    df_data = pd.DataFrame(df_array, columns = pipeline.get_feature_names_out())

    return df_data

def extract_X_y(
        d):
    """
    Extracts X and y data frames
    :param d: dataframe to handle
    :return:
        X : dataframe with covariates
        y : np.array with y values
    """

    X = d.loc[:, [c for c in df_data.columns if c != 'cat__r_label_p']].copy()
    y= d['cat__r_label_p'].astype(int).values

    return X, y

def predict_and_score(
        mydf : pd.DataFrame) -> float:
    """
    builds prediction and returns AUC score for the given dataframe
    :param mydf: dataframe to score
    :return: the AUC score (float
    """
    X, y_true = extract_X_y(mydf)
    y_test_pred = fitted.predict_proba(X)[:, 1]
    score = roc_auc_score(y_true = y_true, y_score = y_test_pred)
    logger.info(f"score: {score}")
    return score


if __name__ == "__main__":

    pipeline_fit = Pipeline([
        ('classifier', XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric='logloss'
        ))
    ])

    SEP = "-"*10
    logger.info(SEP)
    logger.info("Loading data..")
    df_raw = load_data()

    logger.info(SEP)
    logger.info("Featurizing the dataframe")
    df_data = build_featurized_df(df_raw)

    logger.info(SEP)
    logger.info("Splitting the data")
    df_train_val, df_test = train_test_split(df_data, test_size=0.25)
    df_train, df_validation = train_test_split(df_train_val, test_size=0.30)

    logger.info(f"Length of df_train : {(len(df_train))}")
    logger.info(f"Length of df_validation : {(len(df_validation))}")
    logger.info(f"Length of df_test : {(len(df_test))}")

    logger.info(SEP)
    logger.info("Training the model..")
    X_train, y_train = extract_X_y(df_train)
    fitted = pipeline_fit.fit(X_train, y_train)

    logger.info(SEP)
    logger.info("Prediction")

    logger.info("Validation:")
    predict_and_score(df_validation)

    logger.info("Test:")
    predict_and_score(df_test)
