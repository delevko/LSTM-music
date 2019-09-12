from music21 import *
import numpy as np
from matplotlib import pyplot
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, Dropout, Dense, LSTM, Activation
from tensorflow.python.keras import backend, losses, utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import optimizers
import glob
import random

epochs = 65
batch_size = 256
lstm_size = 512
sequence_len = 32

learning_rate = 0.001
dropout = 0.3
r_dropout = 0.0
tempo = 0.25

header = "nottingham/"
i_to_n_file = header + "int_to_note.txt"
train_mapped_file = header + "train_mapped.txt"
test_mapped_file = header + "test_mapped.txt"
weights_file = header + "weights/{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5"

music_header = "nottingham/"
test_folder = "midi/" + music_header + "test/*.mid"
train_folder = "midi/" + music_header + "train/*.mid"
loss_history_file = header + "loss_history.txt"
acc_history_file = header + "acc_history.txt"

weights_to_load = ""


def dict_create(note_list):
    l = len(note_list)
    n_to_i = dict((note_, i) for i, note_ in enumerate(note_list))
    i_to_n = dict((i, note_) for i, note_ in enumerate(note_list))

    i_n = open(i_to_n_file, "w")
    for i in range(l):
        i_n.write(str(i) + ":" + str(note_list[i]) + "\n")
    i_n.close()

    return n_to_i, i_to_n


def input_notes(folder, flag):
    files_list = glob.glob(folder)
    random.shuffle(files_list)

    parsed_notes = []

    for file in files_list:
        notes_to_parse = midi.translate.midiFilePathToStream(file)

        prev_offset = notes_to_parse.flat.notes.elements[0].offset
        
        for n in notes_to_parse.flat.notes.elements:
            if n.offset != prev_offset:
                parsed_notes.append("---")
                
            if isinstance(n, note.Note):
                parsed_notes.append(n.nameWithOctave)
            elif isinstance(n, chord.Chord):
                for p in n.pitches:
                    parsed_notes.append(p.nameWithOctave)
                
            prev_offset = n.offset
        
        parsed_notes.append("---")

        print(file + " parsed")

    return parsed_notes


def map_and_save(n_to_i, note_list, flag):
    parsed = [n_to_i[note_list[i]] for i in range(len(note_list))]

    if flag:
        f = open(train_mapped_file, "w")
    else:
        f = open(test_mapped_file, "w")

    [f.write(str(parsed[i])+"\n") for i in range(len(parsed))]
    f.close()

    return parsed


def seq_prep(list_, dict_):
    input_seq = []
    output_seq = []
    for i in range(len(list_) - sequence_len - 1):
        input_seq.append(list_[i:i+sequence_len])
        output_seq.append(list_[i+sequence_len])

    input_seq = np.reshape(input_seq, (len(input_seq), sequence_len, 1))
    input_seq = input_seq / float(len(dict_))

    output_seq = utils.np_utils.to_categorical(output_seq, len(dict_))

    return input_seq, output_seq


def create_network(network_input, n_vocab):
    print(network_input.shape[0])
    print(network_input.shape[1])
    print(network_input.shape[2])
    print(n_vocab)

    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(lstm_size,
            return_sequences=True, recurrent_dropout=r_dropout),
        input_shape=(network_input.shape[1], network_input.shape[2])
    ))
    model.add(Dropout(dropout))
    model.add(
        Bidirectional(
            LSTM(lstm_size,
            return_sequences=False, recurrent_dropout=r_dropout)
    ))
    model.add(Dropout(dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    optimizer = optimizers.rmsprop()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if weights_to_load != "":
        model.load_weights(weights_to_load)

    return model


def train(model, network_input, network_output, perplexity_input, perplexity_otput, dict_size):
    print("summary:")
    print("dict_size: " + str(dict_size))
    print("net_input shape: " + str(network_input.shape))
    print("net_output shape: " + str(network_output.shape))
    print("perp_input shape: " + str(perplexity_input.shape))
    print("perp_output shape: " + str(perplexity_otput.shape))

    checkpoint = ModelCheckpoint(
        weights_file,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        period=4
    )
    callbacks_list = [checkpoint]

    history = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
                        validation_split=0.15)

    pyplot.plot(history.history["loss"])
    pyplot.plot(history.history["val_loss"])
    pyplot.title("train vs validation loss")
    pyplot.ylabel("loss")
    pyplot.xlabel("epoch")
    pyplot.legend(["train", "validation"], loc="upper right")
    pyplot.savefig("nottinghamsingleU/loss.png")

    pyplot.plot(history.history["acc"])
    pyplot.plot(history.history["val_acc"])
    pyplot.title("train vs validation accuracy")
    pyplot.ylabel("acc")
    pyplot.xlabel("epoch")
    pyplot.legend(["train", "validation"], loc="upper right")
    pyplot.savefig("nottinghamsingleU/acc.png")


if __name__ == "__main__":
    train_notes = input_notes(train_folder, 1)
    test_notes = input_notes(test_folder, 0)
    print("input OK")

    notes_unique = sorted(set(train_notes+test_notes))
    dict_size = len(notes_unique)
    print(str(len(set(train_notes))) + "uniques in train set")
    print(str(len(set(test_notes))) + "uniques in test set")
    print(str(len(notes_unique)) + " unique notes total")

    note_to_int, int_to_note = dict_create(notes_unique)
    dict_size = len(note_to_int)
    print("dict created")

    train_notes = map_and_save(note_to_int, train_notes, 1)
    test_notes = map_and_save(note_to_int, test_notes, 0)
    print("mapped and saved")

    train_input, train_output = seq_prep(train_notes, note_to_int)
    test_input, test_output = seq_prep(test_notes, note_to_int)
    input_size = len(train_input)
    print("input/output prepared")

    model = create_network(train_input, dict_size)
    print("model created")

    train(model, train_input, train_output, test_input, test_output, dict_size)
    print("successfully trained")
