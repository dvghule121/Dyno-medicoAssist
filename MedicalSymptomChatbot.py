import json
import numpy
from keras.models import load_model
import tensorflow as tf

with open("static/data/symtomps.json", 'r') as f:
    intents = json.load(f)


class Dataset:
    def __init__(self):

        self.all_word = []
        doc = []
        treatment = dict()
        self.tags = []
        for intent in intents["intents"]:
            treatment[intent["tag"]] = intent["treatment"]
            for quest in intent["symptoms"]:
                quest = list(quest.split(" "))
                self.all_word.extend(quest)
                doc.append([quest, intent["tag"]])
                if intent["tag"] not in self.tags:
                    self.tags.append(intent["tag"])

        self.all_word = sorted(list(set(self.all_word)))
        output = [0] * len(self.tags)
        X = []
        Y = []

        for docs in doc:
            bag = []
            for i, word in enumerate(self.all_word):
                bag.append(1) if word in docs[0] else bag.append(0)

            output_row = list(output)
            output_row[self.tags.index(docs[1])] = 1
            X.append(bag)
            Y.append(output_row)

        self.train_x = numpy.array(X)
        print(self.train_x.shape)
        self.train_y = numpy.array(Y)
        with open("static/data/treatments.json", 'w') as file:
            json.dump(treatment, file)

    #   ----------------- Predict tag function for keras model --------------------------

    def predict_tag(self, s,model_name="sec_model"):
        """
        function converts user input to model and returns predicted tag
        """
        # loads pretrained keras model
        model = load_model(model_name)

        # data created all words and tags
        all_word = self.all_word
        tags = self.tags

        # creates input data for model
        list_quest_user = [0] * len(self.all_word)

        # replaces 0 by 1 in user query for every word matched in given all_words
        user_input = s
        user_input = user_input.split(" ")
        for word in all_word:
            for q in user_input:
                if q == word:
                    list_quest_user[all_word.index(word)] = 1

        # make predictions and returns max value(probability) of result
        prediction_list = list(model.predict([list_quest_user])[0])

        # convert output(probabilities) to tags
        index = prediction_list.index(max(prediction_list))
        prediction_1 = tags[index]
        prediction_list.pop(index)
        tags.pop(index)

        index = prediction_list.index(max(prediction_list))
        prediction_2 = tags[index]
        prediction_list.pop(index)
        tags.pop(index)

        index = prediction_list.index(max(prediction_list))
        prediction_3 = tags[index]
        prediction_list.pop(index)
        tags.pop(index)

        index = prediction_list.index(max(prediction_list))
        prediction_4 = tags[index]
        prediction_list.pop(index)
        tags.pop(index)

        return [prediction_1, prediction_2,prediction_3,prediction_4]



    # ---------- Old Model creation and fitting input and output data ---------------
    # def fit_model(self, save=True):
    #     """
    #     creates model and fits the data this is keras model with sequential api input layer with size of training data
    #     :returns model # Not needed
    #     """
    #
    #     # Sequential classification model
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape=(len(self.train_x[0]),)),
    #         tf.keras.layers.Dense(64, activation="relu"),
    #         tf.keras.layers.Dense(len(self.train_y[0]), activation="softmax")
    #     ])
    #
    #     model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
    #                   optimizer=tf.keras.optimizers.Adam(0.001),
    #                   metrics=["accuracy"])
    #
    #     model.fit(self.train_x, self.train_y, epochs=100, batch_size=3)
    #
    #     # saving model
    #     if save:
    #         model.save("sec_model")
    #
    #     return model

    # ----------  Model creation and fitting input and output data ---------------

    def fit_model(self, save=True, model_config=None):
        """
        Creates and fits the model using the provided configuration.
        :param save: Boolean indicating whether to save the model
        :param model_config: Dictionary containing model configuration (optional)
        :return: Trained model
        """

        if model_config is None:
            model_config = {
                "input_shape": len(self.train_x[0]),
                "hidden_units": 64,
                "activation": "relu",
                "output_units": len(self.train_y[0]),
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 3
            }

        # Sequential classification model
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(model_config["input_shape"],)),
            tf.keras.layers.Dense(model_config["hidden_units"], activation=model_config["activation"]),
            tf.keras.layers.Dense(model_config["output_units"], activation="softmax")
        ])

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(model_config["learning_rate"]),
                      metrics=["accuracy"])

        # Define callbacks as needed (e.g., ModelCheckpoint, EarlyStopping)

        model.fit(self.train_x, self.train_y, epochs=model_config["epochs"], batch_size=model_config["batch_size"])

        # Saving model
        if save:
            model.save("sec_model")

        return model

    def fit_optimized_model(self, save=True, model_config=None):
        """
        Creates and fits the optimized model using the provided configuration.
        :param save: Boolean indicating whether to save the model
        :param model_config: Dictionary containing model configuration (optional)
        :return: Trained model
        """

        if model_config is None:
            model_config = {
                "input_shape": len(self.train_x[0]),
                "hidden_units": 64,
                "activation": "relu",
                "output_units": len(self.train_y[0]),
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32  # Increased batch size for faster training
            }

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(model_config["input_shape"],)),
            tf.keras.layers.BatchNormalization(),  # Batch Normalization
            tf.keras.layers.Dense(model_config["hidden_units"], activation=model_config["activation"]),
            tf.keras.layers.Dropout(0.5),  # Dropout for regularization
            tf.keras.layers.Dense(model_config["output_units"], activation="softmax")
        ])

        # Implement Learning Rate Scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=model_config["learning_rate"],
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=optimizer,
                      metrics=["accuracy"])

        # Implement Early Stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(self.train_x, self.train_y, epochs=model_config["epochs"],
                            batch_size=model_config["batch_size"],
                            validation_split=0.2,  # Using validation split
                            callbacks=[early_stopping])

        # Saving model
        if save:
            model.save("optimized_sec_model")

        return model, history

    # ------ Function to return treatments of predicted tag-------------------

    @staticmethod
    def treat(tag):
        """
        :param tag:
        :returns Treatments and precautions for disease
        """
        # loads Treatments file
        with open("static/data/treatments.json", 'r') as file:
            treatment = json.load(file)
        return treatment[tag][0]

    # ---------- converts keras model to tflite model for faster prediction on low end devices

    def tflite_converter(self,model):
        """
        takes keras model and converts to tflite
        :param self:
        :param model:
        :return:
        """
        # def tensor flow converter
        converter = tf.lite.TFLiteConverter.from_saved_model(model)  # path to the SavedModel directory
        tflite_model = converter.convert()

        # Save the model.
        with open('static/data/model.tflite', 'wb') as f:
            f.write(tflite_model)




    def  return_symp(self, name):
        symptoms = ''
        for i in intents['intents']:
            if i['tag'] == name:
                symptoms = i['symptoms'][0]
                break
        return symptoms





if __name__ == '__main__':
    p = Dataset()
    # model = p.fit_model()
    model2 = p.fit_optimized_model()
    print(p.predict_tag("buzzing in ears","optimized_sec_model"))
    # print(p.treat("common_cold"))
    # p.tflite_converter('sec_model')
    # print(p.predict_tag("cough fever sneezing headache tiredness") )
