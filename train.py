import json
import numpy
"https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-win_amd64.whl#sha256=414f4a2e2d141d812507f0055591f50d65e714e0a4501ad1c43b02108419d025"



import tflite_runtime.interpreter as tflite


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
    """
    def predict_tag(self, s,model_name="sec_model"):
        \"\"\"
        function converts user input to model and returns predicted tag
        \"\"\"
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
        return tags[prediction_list.index(max(prediction_list))]
        
    """

    # ---------- Model creation and fitting input and output data ---------------
    """
    def fit_model(self,save=True):
        \"\"\"
        creates model and fits the data this is keras model with sequential api input layer with size of training data
        :returns model # Not needed
        \"\"\"

        # Sequential classification model  
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(len(self.train_x[0]),)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(len(self.train_y[0]), activation="softmax")
        ])

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=["accuracy"])

        model.fit(self.train_x, self.train_y, epochs=50, batch_size=45)
        
        # saving model 
        if save:
            model.save("sec_model")

        return model
    """

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
    """
    def tflite_converter(self,model):
        \"\"\"
        takes keras model and converts to tflite
        :param self: 
        :param model:
        :return:
        \"\"\"
        # def tensor flow converter
        converter = tf.lite.TFLiteConverter.from_saved_model(model)  # path to the SavedModel directory
        tflite_model = converter.convert()

        # Save the model.
        with open('static/data/model.tflite', 'wb') as f:
            f.write(tflite_model)
    """

    #   ----------------- Predict tag function for tflite model --------------------------

    def predict_tag(self, s):
        """
        predicts tags faster from lite model
        :param s:
        :return:
        """

        all_word = self.all_word
        tags = self.tags

        list_quest_user = [0] * len(self.all_word)

        user_input = s
        user_input = user_input.split(" ")
        for word in all_word:
            for q in user_input:
                if q == word:
                    list_quest_user[all_word.index(word)] = 1
        list_quest_user = numpy.array(list_quest_user, dtype=numpy.float32)

        tflite_interpreter = tflite.Interpreter(model_path="static/data/model.tflite")

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        # ------------------ to check input and output shape  ----------------------
        # print("== Input details ==")
        # print("shape:", input_details[0]['shape'])
        # print("type:", input_details[0]['dtype'])
        # print("\n== Output details ==")
        # print("shape:", output_details[0]['shape'])
        # print("type:", output_details[0]['dtype'])
        # ---------------------------------------------------------------------------

        # --------------------- Output will look like this --------------------------
        # >> == Input details ==
        # >> shape: [  1 224 224   3]
        # >> type: <class 'numpy.float32'>
        # >> == Output details ==
        # >> shape: [1 5]
        # >> type: <class 'numpy.float32'>
        # ---------------------------------------------------------------------------

        # ------------------  Resizes the input shape if needed  ------------------
        tflite_interpreter.resize_tensor_input(input_details[0]['index'], (1, 684))
        tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 62))
        tflite_interpreter.allocate_tensors()

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        #
        # ----------------- input and output shapes after Resizing ------------------
        # print("== Input details ==")
        # print("shape:", input_details[0]['shape'])
        # print("\n== Output details ==")
        # print("shape:", output_details[0]['shape'])

        # --------------------- Output looks  like this -------------------------------
        # >> == Input details ==
        # >> shape: [ 1 664]
        # >> == Output details ==
        # >> shape: [1  62]

        # Set batch of images into input tensor
        tflite_interpreter.set_tensor(input_details[0]['index'], [list_quest_user])
        # Run inference
        tflite_interpreter.invoke()
        # Get prediction results
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        # print("Prediction results shape:", tflite_model_predictions.shape)
        # print(tflite_model_predictions)

        # convert output(probabilities) to tags
        prediction_list = list(tflite_model_predictions[0])
        return tags[prediction_list.index(max(prediction_list))]
    




if __name__ == '__main__':
    p = Dataset()
    # model = p.fit_model()
    # print(p.predict_tag("cough fever sneezing headache tiredness high temperature","sec_model"))
    # print(p.treat("common_cold"))
    # p.tflite_converter('sec_model')
    print(p.predict_tag("cough fever sneezing headache tiredness"))
