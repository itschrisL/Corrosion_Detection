# CorrosionDetection.

Authors: Jacob Apenes, Chris Lytle


# How to run the application 

1) Include Images of corrosion that you want to use to train or classify in the directory 'Images'.  Then make two
 sub-directories for the original images and the ground truth images.
2) Then in the method Main, replace the directory names of 'Original_Images' and 'GT_Images' with the names of the two 
directories tha you have just created in step one.  The main method is located at the bottom of the 'main.py' file.  
3) After that you can run the main method with python3.  

Note: You may need to install the imported libraries to run the program, like TensorFlow or NumPy.  

# Training the Model

The code is initially setup to run the training process automatically with the specified images.  It also uses the 
already trained weights by default when training and updates them when training.  

In main, if you comment out the line for training process you can skip it and just skip to the prediction part.  
You can also choose to use new weights when training by changing the use_cp parameter in the training method called in
main.  

The main file is used for pre processing other helper methods.  Then the file called Corrosion_Detection_Model is used 
to create and train the model.  It is a separate class. 
   



