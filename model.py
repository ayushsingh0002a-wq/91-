from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(df):
    X = df[['number']]
    y_size = df['size']
    y_color = df['color_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y_size, test_size=0.2)

    size_model = RandomForestClassifier()
    size_model.fit(X_train, y_train)

    color_model = RandomForestClassifier()
    color_model.fit(X_train, y_color)

    return size_model, color_model


def predict_next(size_model, color_model, last_number):
    size_pred = size_model.predict([[last_number]])[0]
    color_pred = color_model.predict([[last_number]])[0]

    size = "Big" if size_pred == 1 else "Small"

    color_map_rev = {0: 'Red', 1: 'Green', 2: 'Violet'}
    color = color_map_rev[color_pred]

    return size, color
