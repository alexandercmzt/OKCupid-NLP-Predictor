Alexander Chatron-Michaud
260611509

Code for final project for COMP-599

*Note: The code here provides the main scripts used to run this project, for the purpose of making grading easier. The full project contains 250GB of files to run 
including all the data, models, unorganized sub-scripts, etc, so this repo contains the main parts of the code that was run as evidence. Getting the GPU to 
work for this code is also quite a long task. I hope that the code provided is legible enough and clear enough to demonstrate the process by which the project was run. 

classifiers/ holds code for running classifiers (this code seems small but it has been changed massively over time. also add weeks of training time for larger LSTMs...)
preprocessing/ holds code for loading text data and formatting then dumping to disk
data/ holds the data (email me if you want it, but be warned, it's about 200GB)
written/ holds all the written work for this project i.e. proposal, paper submission

Please contact me if there are questions about the code. You need these dependencies to run it

Bleeding edge/development edition for:
sklearn
theano
tensorflow
keras
gensim
numpy
scipy
cPickle
CUDA architectures - GPU must be set up as GPU1 in your configuration for tensorflow backend.

Methods and blocks of code have comments explaining what they do. Most of the work was done in IPython however, using the scripts as reference. I have tried running 
this on multiple computers and some computers will take ~half an hour to set up.


