
from sklearn.model_selection import train_test_split


def transform(df):
    """
    Split the Iris dataset into 80% training
    data and 20% testing data.
    """

    output = df.copy()

    train_indices, test_indices = train_test_split(
        output.index,
        test_size=0.3,
        random_state=42,
        stratify=output["target"]
    )

    output["split"] = "train"
    output.loc[test_indices, "split"] = "test"

    return output
