<p>Tensorflow.Keras project for music generation using LSTM</br>
My example works on Nottingham music dataset (MIDI format):</br>
https://github.com/jukedeck/nottingham-dataset
</p>

<p>Link to the original idea: </br>https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5</p>
</br>


<p>How to train your model:</br>
Put your training dataset containing MIDI files in midi/nottingham/train/ folder</br>
Same for validation dataset (midi/nottingham/test/)</br>

<code>python3 train.py</code> will start training of your model, all hyperparameters can be changed in the source code</br>
Weights (.hdf5 format) are stored in nottingham/weigths/ folder only if improvements are reached
Also loss and accuracy of learning epoch by epoch will be stored in nottingham/results/ folder
</p>

<p>After training(might take few hours) you can generate your music samples from the model by specifying your favorite weights file:
<code>python3 generate.py &lt;weightsFile></code></br>
Generated samples are stored in .mid format in nottingham/ folder

<p>Requirements:</br>
CUDA toolkit + compatible GPU</br>
Tensorflow + Keras</br>
python3, numpy, matplotlib</br>
music21</br>
</p>

<p>Best environment setup tutorials (IMHO):</br>
<b>Windows:</b></br>
https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10</br>

<b>Linux:</b></br>
https://www.tensorflow.org/install/gpu</br>
https://www.tensorflow.org/install/pip</br>
<b>OR</b></br>
https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070
</p>

<p></p>
<p>Best books describing techniques and challenges in ML music generation (again IMHO):</br>
http://people.idsia.ch/~juergen/blues/IDSIA-07-02.pdf</br>
https://arxiv.org/abs/1709.01620
</p>
