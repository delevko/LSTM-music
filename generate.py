import pickle
import numpy
from music21 import *
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional
from tensorflow.python.keras import optimizers

lstm_size = 512
sequence_len = 32

learning_rate = 0.001
dropout = 0.3
r_dropout = 0.0
tempo = 0.25

rangelen = 250

header = "nottingham/"
weightsfile = sys.argv[1]
# header + "weights/" + "64-0.0880-0.9716.hdf5"
train_mapped_file = header + "train_mapped.txt"
i_to_n_file = header + "int_to_note.txt"
sample_file = header + "out3.mid"


def get_notes():
    f = open(train_mapped_file, "r")
    line = f.readline()
    notes = []
    while line:
        notes.append(int(line[:-1]))
        line = f.readline()

    return notes


def prepare_sequences(notes, n_vocab):
    input_seq = []

    for i in range(len(notes) - sequence_len - 1):
        input_seq.append(notes[i:i+sequence_len])
    input_n = len(input_seq)

    norm_input = input_seq
    norm_input = numpy.reshape(norm_input, (input_n, sequence_len, 1))
    norm_input = norm_input / float(n_vocab)

    return input_seq, norm_input


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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    # Load the weights to each node
    print(weightsfile)
    model.load_weights(weightsfile)

    return model


def generate_notes(model, network_input, int_to_note, n_vocab, seed):
    
    if seed == "":
        start = numpy.random.randint(0, len(network_input)-1)
        pattern = network_input[start]
        print(pattern)
    else:
        # for consecutive verses pick previous as a seed
        pattern = seed
    
    prediction_output = []
    counter = 0

    while counter < 2*32:
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = numpy.argmax(prediction)

        print(prediction[0][index])
        result = int_to_note[index]
        if result == "---":
            counter += 1

        prediction_output.append(result)
        
        pattern = pattern[1:]
        pattern.append(index)

    return prediction_output, pattern


def decode_notes(note_list):
    notes_to_parse = []
    offset = 0.0

    chrd = chord.Chord()
    for elem in note_list:
        if elem == "---":
            if chrd.pitches != ():
                chrd.offset = offset
                if len(chrd.pitches) == 1:
                    chrd.duration.quarterLength = 0.5
                else:
                    chrd.duration.quarterLength = 1.0

                notes_to_parse.append(chrd)
                
                chrd = chord.Chord()
                offset += tempo

        else:
            chrd.add(note.Note(nameWithOctave=elem))

    return notes_to_parse


def get_int_to_note():
    f = open(i_to_n_file, "r")
    line = f.readline()
    i_to_n = {}

    while line:
        elem = line.split(':')
        i_to_n[int(elem[0])] = str(elem[1][:-1])
        line = f.readline()

    return i_to_n


if __name__ == '__main__':
    notes = get_notes()
    int_to_note = get_int_to_note()
    n_vocab = len(set(int_to_note))

    network_input, normalized_input = prepare_sequences(notes, n_vocab)
    model = create_network(normalized_input, n_vocab)
    
    verse1, v1_pattern = generate_notes(model, network_input, int_to_note, n_vocab, "")
    verse2, v2_pattern = generate_notes(model, network_input, int_to_note, n_vocab, v1_pattern)
    verse3, _ = generate_notes(model, network_input, int_to_note, n_vocab, v2_pattern)

    chorus, _ = generate_notes(model, network_input, int_to_note, n_vocab, "")

    prediction_output = (verse1+chorus + verse2+chorus + verse3+chorus)
    print(prediction_output)
    decoded = decode_notes(prediction_output)

    SS = stream.Stream(decoded)
    SS.write('midi', fp=sample_file)
    # SS.show('midi')
