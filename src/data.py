from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# The 13 chemical measurements, in the order the model expects them. The dataset calls the
# 12th one "od280/od315_of_diluted_wines"; the slash is replaced so it can be an API field name.
FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]

# The three grape varieties (cultivars) the wines were made from
CLASS_NAMES = ["cultivar_1", "cultivar_2", "cultivar_3"]


def load_data():
    """
    Load the wine dataset that ships with scikit-learn (178 wines from one region of Italy).
    Returns:
        X (pandas.DataFrame): The 13 measurements, with the column names in FEATURES.
        y (pandas.Series): The cultivar of each wine, as a name from CLASS_NAMES.
    """
    wine = load_wine(as_frame=True)
    X = wine.data.rename(columns={"od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"})
    y = wine.target.map(dict(enumerate(CLASS_NAMES)))
    return X[FEATURES], y


def split_data(X, y):
    """
    Split the data into training and testing sets, keeping the same mix of cultivars in each.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)
