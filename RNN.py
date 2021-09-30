from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout
from keras.layers.embeddings import Embedding
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import collections
import pandas as pd
import numpy as np
import os
import json


class RNNModel:

    def __init__(self):
        self.in_data = None
        self.out_data = None
        self.preproc_in = None
        self.preproc_out = None
        self.in_tk = None
        self.out_tk = None
        self.model = None
        self.df_length = 0
        self.percentage_true_neg = 0
        self.infos = None
        self.to_predict = ""
        self.predicted = ""

    def get_to_predict(self):
        return self.to_predict

    def get_predicted(self):
        return self.predicted

    def set_percent(self, perc):
        self.percentage_true_neg = perc

    def get_infos_obj(self):
        return self.infos

    def get_infos(self):
        in_words_counter = collections.Counter([word for sentence in self.in_data for word in sentence.split()])
        out_words_counter = collections.Counter([word for sentence in self.out_data for word in sentence.split()])

        self.preproc_in, self.preproc_out, self.in_tk, self.out_tk = self.preprocess(self.in_data, self.out_data)
        max_in_length = self.preproc_in.shape[1]
        max_out_length = self.preproc_out.shape[1]
        in_vocab_size = len(self.in_tk.word_index)
        out_vocab_size = len(self.out_tk.word_index)

        # log messages
        print("Informations about dataset (in/out)")

        # in data
        print("===================================")
        print("==========\nInput data:\n==========")
        print("===================================")
        print(f"Total words: {len([word for sentence in self.in_data for word in sentence.split()])}")
        print(f"Unique words: {len(in_words_counter)}")
        # print(f"10 most commun words: {'" "'.join(list(zip(*in_words_counter.most_common(10))))}")
        print(f"Max input sentence length: {max_in_length}")
        print(f"Input vocabulary size: {in_vocab_size}")

        # out data
        print("====================================")
        print("==========\nOutput data:\n==========")
        print("====================================")
        print(f"Total words: {len([word for sentence in self.out_data for word in sentence.split()])}")
        print(f"Unique words: {len(out_words_counter)}")
        # print(f"10 most commun words: {'" "'.join(list(zip(*out_words_counter.most_common(10))))}")
        print(f"Max output sentence length: {max_out_length}")
        print(f"Output vocabulary size: {out_vocab_size}")

        # store infos in object
        self.infos = [
            {
                "io": "input",
                "total_words": len([word for sentence in self.in_data for word in sentence.split()]),
                "unique_words": len(in_words_counter),
                "max_io_sentence_length": max_in_length,
                "io_voc_size": in_vocab_size
            }, {
                "io": "output",
                "total_words": len([word for sentence in self.out_data for word in sentence.split()]),
                "unique_words": len(out_words_counter),
                "max_io_sentence_length": max_out_length,
                "io_voc_size": out_vocab_size
            }
        ]

        # log message
        print("End of infos... \n[next step]\n")

    @staticmethod
    def tokenize(x):
        x_tk = Tokenizer()
        x_tk.fit_on_texts(x)
        return x_tk.texts_to_sequences(x), x_tk

    @staticmethod
    def pad(x, length=None):
        if length is None:
            length = max([len(sentence) for sentence in x])
        return pad_sequences(x, maxlen=length, padding='post', truncating='post')

    def preprocess(self, x, y):
        preprocess_x, x_tk = self.tokenize(x=x)
        preprocess_y, y_tk = self.tokenize(x=y)

        preprocess_x = self.pad(preprocess_x)
        preprocess_y = self.pad(preprocess_y)

        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

        return preprocess_x, preprocess_y, x_tk, y_tk

    @staticmethod
    def logits_to_text(logits, tokenizer):
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'
        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    def embed_model(self, input_shape, output_sequence_length,
                    in_vocab_size, out_vocab_size, learning_rate,
                    gru_neurons, dropout_value, dense_neurons,
                    loss_function, gru_neuron):

        learning_rate = learning_rate
        model = Sequential()
        model.add(Embedding(in_vocab_size, 100, input_length=input_shape[1], input_shape=input_shape[1:]))
        model.add(GRU(gru_neurons, return_sequences=True))
        model.add(Dropout(dropout_value))
        model.add(GRU(gru_neurons, return_sequences=True))
        model.add(Dropout(dropout_value))
        model.add(TimeDistributed(Dense(dense_neurons, activation='relu')))
        model.add(Dropout(dropout_value))
        model.add(TimeDistributed(Dense(out_vocab_size, activation='softmax')))

        model.compile(
            loss=sparse_categorical_crossentropy,
            optimizer=Adam(learning_rate),
            metrics=['accuracy']
        )

        self.model = model

        # save model.summary()
        filename = "model_"+str(self.df_length)+".h5"
        with open("./analysis/compar_epochs/"+str(self.df_length)+"/"+filename, "w") as model_file:
            model.summary(print_fn=lambda x: model_file.write(x + '\n'))

        # save ploted model
        plot_model(
            model,
            to_file="./analysis/compar_epochs/"+str(self.df_length)+"/model_"+str(self.df_length)+"_"+str(gru_neuron)+".png",
            show_shapes=True,
            show_layer_names=True
        )

        return model

    @staticmethod
    def get_init_sequence(logits, word_index):
        init_sequence = []
        logits = logits.tolist()
        values = [value for value in word_index.values()]
        keys = [key for key in word_index.keys()]
        for logit in logits:
            for i in range(len(values)):
                if int(logit) == int(values[i]):
                    init_sequence.append(keys[i])
        return ' '.join(init_sequence)

    def model_summary(self):
        return self.model.summary()

    def run(self, nb_epochs, batch_size, validation_split, dataset_path,
            gru_neurons, learning_rate, dropout_value, dense_neurons, loss_function, gru_neuron):

        # read csv
        # TODO: check if dataset_path[-4]==csv
        df = pd.read_csv(dataset_path)
        in_data = [item for item in df['in']]
        out_data = [item for item in df['out']]

        # store data in self object
        self.in_data = in_data
        self.out_data = out_data

        # preproc_in/out
        self.get_infos()

        # store df length
        if len(self.in_data) == len(self.out_data):
            self.df_length = len(self.in_data)
        else:
            self.df_length = max(len(self.in_data), len(self.out_data))

        # log message
        print("====================\n>>> Dataset loaded !\n====================\n[next step]")

        # rnn
        tmp_x = self.pad(self.preproc_in, self.preproc_out.shape[1])
        tmp_x = tmp_x.reshape((-1, self.preproc_out.shape[-2]))

        embed_rnn_model = self.embed_model(
            input_shape=tmp_x.shape,
            output_sequence_length=self.preproc_out.shape[1],
            in_vocab_size=len(self.in_tk.word_index) + 1,
            out_vocab_size=len(self.out_tk.word_index) + 1,
            learning_rate=learning_rate,
            gru_neurons=gru_neurons,
            dropout_value=dropout_value,
            dense_neurons=dense_neurons,
            loss_function=loss_function,
            gru_neuron=gru_neuron
        )

        history_const = embed_rnn_model.fit(
            tmp_x,
            self.preproc_out,
            batch_size=batch_size,
            epochs=nb_epochs,
            validation_split=validation_split
        )

        # log message
        print("========== End of current epoch ==========")

        # store hyperparameters
        history_const.history['epochs'] = nb_epochs
        history_const.history['df_length'] = self.df_length
        history_const.history['percentage_true_neg'] = self.percentage_true_neg
        history_const.history['batch_size'] = batch_size

        # store vocab infos
        buffer_voc_in = self.infos[0]
        buffer_voc_out = self.infos[1]
        history_const.history['input_vocab'] = {
            "total_words": buffer_voc_in['total_words'],
            "unique_words": buffer_voc_in['unique_words'],
            "max_in_sentence_length": buffer_voc_in['max_io_sentence_length'],
            "in_voc_size": buffer_voc_in['io_voc_size']
        }
        history_const.history['output_vocab'] = {
            "total_words": buffer_voc_out['total_words'],
            "unique_words": buffer_voc_out['unique_words'],
            "max_in_sentence_length": buffer_voc_out['max_io_sentence_length'],
            "in_voc_size": buffer_voc_out['io_voc_size']
        }

        # log message
        print("================================================")
        print("========== Current Training Completed ==========")
        print("================================================")

        directory = "ep" + str(nb_epochs) + "-dflgt" + str(self.df_length) + "-perc" + str(
            int(self.percentage_true_neg)) + "-btchsz" + str(batch_size)
        path = "histories/"+str(self.df_length)+"/" + directory

        if not os.path.exists(path):
            os.mkdir(path)

        path_model = path + "/m--" + directory + ".h5"

        embed_rnn_model.save(path_model)

        # log message
        print(f"Current model is saved in {path_model}")

        self.to_predict = self.get_init_sequence(tmp_x[:1][0], self.out_tk.word_index)
        self.predicted = self.logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0], self.out_tk)

        history_const.history['init_seq'] = "init sequence"
        history_const.history['to_predict'] = self.to_predict
        history_const.history['prediction'] = self.predicted

        filename = path + "/f--" + directory + ".json"

        # save history
        with open(filename, 'w') as outfile:
            json.dump(history_const.history, outfile, indent=3)