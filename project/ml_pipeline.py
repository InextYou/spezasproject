import pandas as pd
import numpy as np


class MLPipeline:

    #------------- pandas set up ----------#
    pd.set_option('display.width', 600)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 170)
    # -------------------------------------#

    def __init__(self):
        # Load data
        df = self.load_data()
        # Preprocessing of data
        x, y = self.split_df_to_xy(df)
        x = self.normalize(x)
        # Split test/train datset
        x_train, x_test, y_train, y_test = self.split_into_train_test(x, y)
        # One hot encode y
        y_train = self.one_hot_encode(y_train)
        y_test = self.one_hot_encode(y_test)
        # Build Model and train it
        mbt = self.ModelBuilderAndTrainer()
        res = {}
        for j in range(10):
            model = mbt.build(number_of_hidden_layers=j, number_of_units_per_layer=5)
            trained_model, history = mbt.train(model, x_train, y_train, batch_size=64, epochs=10)
            # Test model

            y_pred = model.predict(x_test)
            # Converting predictions to label
            pred = list()
            for i in range(len(y_pred)):
                pred.append(np.argmax(y_pred[i]))
            # Converting one hot encoded test label to label
            test = list()
            for i in range(len(y_test)):
                test.append(np.argmax(y_test[i]))

            from sklearn.metrics import accuracy_score
            a = accuracy_score(pred, test)
            #print('Accuracy is:', a * 100)
            res.update({j: float(a)})
            del model

        print(res)
        import matplotlib.pyplot as plt
        plt.plot(list(res.keys()), list(res.values()))
        plt.show()


    @staticmethod
    def load_data():
        path_to_file = "../data/Component_Faults_Data.csv"
        df = pd.read_csv(path_to_file)
        return df

    @staticmethod
    def split_df_to_xy(df):
        number_of_rows = df.shape[1]
        x = df.iloc[:, :48].values
        y = df["class"].values.reshape(-1, 1)
        return x, y

    @staticmethod
    def normalize(x):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x = sc.fit_transform(x)
        return x

    @staticmethod
    def one_hot_encode(y):
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y).toarray()
        return y

    @staticmethod
    def split_into_train_test(x, y):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        return x_train, x_test, y_train, y_test

    class ModelBuilderAndTrainer:

        # params for building model
        input_layer_dim = 48
        output_layer_dim = 11

        def build(self, number_of_hidden_layers, number_of_units_per_layer,
                  activation_function='relu', loss_function='categorical_crossentropy', optimizer='sgd'):

            from keras.models import Sequential
            from keras.layers import Dense

            model = Sequential()
            # Input layer
            model.add(Dense(number_of_units_per_layer, input_dim=self.input_layer_dim, activation=activation_function))
            # Hidden layer
            for i in range(number_of_hidden_layers):
                model.add(Dense(number_of_units_per_layer, activation=activation_function))
            # Output layer
            model.add(Dense(self.output_layer_dim, activation='softmax'))

            model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

            return model

        @staticmethod
        def train(model, x_train, y_train, epochs, batch_size):
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
            return model, history

