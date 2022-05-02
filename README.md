# voice-to-text-AI-model
in this project we aim to learn how to build a model to convert the command voice to text and save it in file .

Ai can’t deal with the sound waves so if we need to recognize the
voice we will convert it to the wave form and draw it .
But in this case we will have a problem all waves in time domain are
very similar to each other .
So we have a frequency domain we can use it but we have another
problem in this case if we have some noise the word which you say
will be another word in the model and the number of frequency are
very huge .
What if we draw every number of this frequency by a specific color so
we can use Mel frequency cepstral coefficient ( Mfcc ) .

![image](https://user-images.githubusercontent.com/57018506/166321360-e451b81f-0330-4ef3-83dc-4b51ebd50fb7.png)
##Mel frequency cepstral coefficient
##(MFCC)
This is the most important thing to make us recognize the speech .
because this tool make us to convert the sound wave to image
contain all the frequency and every specific frequency have a specific
color .
Before we make a model to recognize the speech we need a lot o
data to give it to the model to train by it .
![image](https://user-images.githubusercontent.com/57018506/166322418-e18b8c4a-d3a3-4aca-bdfa-17b67ed895ba.png)

##Preparing the data 
This the first step to build the speech recognition model .
In this step we need some requirement
1- Python 3.8
2- Json library
3- Librosa library
4- Os library
To make a good accuracy you want to have a lot of data so in this
model we will predict the English numbers from 0 to 9 .
So we collect 2850 audio file for each number to train the model by it
.
To use the library which help you in this code we will import it first .
We should read the whole audio file for all number so we use os
library for that .
We need to convert every audio file to MFCC form to make it easy for
CNN take some features from every audio file to train the model by it
so librosa will help us for that .
Now we will read all audio file and convert it to MFCC form .
Now we ready for preparing the data .
We will read all audio file and convert it to MFCC form and we will
make a dictionary that contain (mapping , labels , MFCCs , files )
Mapping contain the name of the directory which have the 2800 audio
file so in this case mapping have 10 word from zero to nine .
Labels contain number that pointing to every number . in this case the
first number pointing to zero and so on .
MFCCS pointing to the number of MFCC for each audio file .
File pointing to the name file of every number.
We will load the whole audio file by librosa and convert it to MFCC
and after we do that we will save this dictionary in json file .
When we load the audio file and convert it we give it some arguments
Num_Mfcc → Number of coefficients to extract . 
N_fft → Interval we consider to apply FFT. Measured in # of samples.
hop_length → Sliding window for FFT. Measured in # of samples .
in this data we will load the audio file that contain 1 second of voice
only so SAMPLES_TO_CONSIDER = 22050 this number mean 1
sec of speech .
we need to save all this data in a json file because this type is make
the read data is very easy to deal with it so we use the third library
json library .
now we are ready to build a model by our own data which we
preparing above .

##Building CNN model
![image](https://user-images.githubusercontent.com/57018506/166322718-4663b5ac-0365-4893-8b66-10e79d14f7b3.png)

The requirements in the 2nd step are
1- json library
2- Tensorflow library ( try to install it by anaconda because this
library will make some problem if you try to install it by the
terminal by pip so I recommend to download anaconda and use
this command ~ conda install tensorflow , and you can do it with
the other library by this command ~ conda install lib_name )
3- Numpy library
4- matplotlib.pyplot library
5- from sklearn.model_selection import train_test_split

I tried to use vosk library to build a model but I found it very difficult to
install and caused different problems so I used these libraries .
We start by reading the json file that we made in the 1st step .
And store the MFCC and Labels in x and y variables.

20% of data we will use it to tset the model and the other 80% we will
use it to train the model .

X_train is 2d array and we should convert it to 3d array so we use
numpy to do this step and we put the third layer by 1 to make the
image gray and we do this step to the x_test and x_validation .


after that we make a number of convolution layers in this model we
make 3 layers but we can make more or less number if we want . it's
depend on the accuracy so we can say number of layers is a
hyperparameter .
for each layer we out the 1st number which pointing to number of
filters we use so in the first layer we use 64 filter and we use
activation function " relu " and this is relu function f(x)=max(0,x)

![image](https://user-images.githubusercontent.com/57018506/166322929-8492f6ef-c06f-472c-a126-5bd1a425b345.png)

##activationfunction

activation functions give out the final value given out from a neuron.
So, an activation function is basically just a simple function that
transforms its inputs into outputs that have a certain range. There
are various types of activation functions that perform this task in a
different manner, For example, the sigmoid activation function
takes input and maps the resulting values in between 0 to 1.
One of the reasons that this function is added into an artificial
neural network in order to help the network learn complex patterns
in the data. These functions introduce nonlinear real-world
properties to artificial neural networks. Basically, in a simple neural
network, x is defined as inputs, w weights, and we pass f (x) that is
the value passed to the output of the network. This will then be the
final output or the input of another layer.
If the activation function is not applied, the output signal becomes a
simple linear function. A neural network without activation function
will act as a linear regression with limited learning power. But we
also want our neural network to learn non-linear states as we give it
complex real-world information such as image, video, text, and
sound.

##Max pooling
Pooling layers are used to reduce the dimensions of the feature
maps. Thus, it reduces the number of parameters to learn and the
amount of computation performed in the network.
The pooling layer summarises the features present in a region of
the feature map generated by a convolution layer. So, further
operations are performed on summarised features instead of
precisely positioned features generated by the convolution layer.
This makes the model more robust to variations in the position of
the features in the input image.
Types of Pooling Layers
1- Max pooling ( recommended in many models )
is a pooling operation that selects the maximum element from
the region of the feature map covered by the filter. Thus, the
output after max-pooling layer would be a feature map
containing the most prominent features of the previous feature
map.

![image](https://user-images.githubusercontent.com/57018506/166323110-0f272696-5346-4e20-8432-a32ece51ca0f.png)

2- Average Pooling
Average pooling computes the average of the elements present
in the region of feature map covered by the filter. Thus, while
max pooling gives the most prominent feature in a particular
patch of the feature map, average pooling gives the average of
features present in a patch.

![image](https://user-images.githubusercontent.com/57018506/166323181-ea4bf19d-e42b-4ef8-a6e6-9d0e19020d3e.png)


In the 2nd convolution layer we do the same steps but we reduce the
number of filters to 32 .
In the 3rd convolution layer we do the same steps but we make the
kernel of maxpooling (2 , 2) .

![image](https://user-images.githubusercontent.com/57018506/166323308-afa15b15-309e-4aa9-8b45-2d0875408149.png)

Now we will make a flatting step to make the output reshape in 1D
array to make it a input to the last neurons to give us the nearby
output .
Dropout used to reduce the number of the neurons which we don't
need because there are not important so that make the model make
processing faster .
to reduce the processing of model again we need to make a softmax
activation to make the model faster and faster .
optimizer help model to detect the final value of input we user the
common optimizer " adam " .

we use a variable that called epochs that mean the number of the
model will see the whole data and that variable is a hyperparameter
and in this model we put it 40 .
batch_size → Samples per batch
patience → Num epochs to wait before early stop, if there isn't an
improvement on accuracy.
to avoid the processing doesn't give us a improvement in the
accuracy we use Earlystopping when the change in delta loss is
0.001 and in this model we use "sparse_categorical_crossentropy" as a loss function

##summary of model
![image](https://user-images.githubusercontent.com/57018506/166323558-19da14de-5c9e-4983-9e2f-984ef58ff443.png)

##plot the model

![image](https://user-images.githubusercontent.com/57018506/166323685-8488b8c2-4579-40b9-ab46-cc22d6eae699.png)


##Recording
We need to record our voice and make a prediction for this audio file .
The requirement for this step
1- Pyaudio library
2- Wave library
Pyaudio library make the recording is very easy

We record the voice for only 2 second because in our model we use
a data contain a 1 second voice for each audio file so in this we
record for 2 second and take the voice in middle of this audio file to
avoid the noise and the silence .
After we record the voice we save it by output.wave file

##Prediction
Finally we reach to the final step in this model now we can record a
voice and we have a model that contain data cam make us to detect
the word that we say
The requirement of this step
1- Librosa library
2- Tensorflow library
3- numpy library

first step is to load the mapping that we are made in the first step of
this mode .
and then we take the path of the audio file that we save in the 3rd step
and load it by librosa library and get the MFCC for it .
and make it a 3d array by putting new axis that make it a gray image
by using numpy .
before that we should ensure that audio file is 1 sec and if it large
than 1 second we take the middle 1 second of this audio file .

then we send this MFCC data to the model so the model will compare
the data which sent with the data which stored in the model .
after that the model will get the mapping word that compatible with
the data of MFCC and put it in keyword variable .

in the end we write the word which the model predicted in a file
because if anyone need to use this word as a speech command
service he will read it from the file and do what he wan

##Online model

if you can provide a network in your application you can use a google
assistant model by using this code 

You need to install the speech_recognition library . then this code is
recording your voice and send it to google assistant model and this
model will know the whole words that you talk and send it back in a
string form .
This is the online solution for speech recognition problem .

you will find summary pdf file above . in this file you will find all code sections with the explanation . 
