
class LogisticRegression():
    def __init__(self, max_iter=50, multi_class='multinomial'):
        self.max_iter = max_iter
        self.multi_class = multi_class
    def fit(self, features_train, fetal_health_train):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=self.max_iter, multi_class=self.multi_class)
        self.model.fit(features_train, fetal_health_train)
    def predict(self, features):
        return self.model.predict(features)


