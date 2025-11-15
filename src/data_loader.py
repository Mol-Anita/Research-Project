import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class URLPhishDataLoader:
    def __init__(self, csv_path: str, scale_features: bool = False):
        self.csv_path = csv_path
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None

    def load_data(self):
        df = pd.read_csv(self.csv_path)

        # Drop reference columns
        df.drop(columns=['url', 'dom', 'tld'], inplace=True, errors='ignore')

        # Separate features and target
        X = df.drop('label', axis=1)
        y = df['label']

        # Initial train/test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        # Further split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.12, stratify=y_train_val, random_state=42
        )

        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test
