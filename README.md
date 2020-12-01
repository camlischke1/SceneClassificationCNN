# SceneClassificationCNN
This uses the Scene Classification Kaggle dataset. The task is to identify which kind of scene can the image be categorized into:
0: Buildings, 1: Forests, 2: Mountains, 3: Glacier, 4: Sea, 5: Street.

# Data Preprocessing
Be sure to download the Kaggle dataset named Scene Classification. Then with train.csv and test.csv in the parent directory, run the file named *SaveToNumpy.py* to preprocess and
save the preprocessed data to numpy arrays. When executing, the *SaveToNumpy.py* file will build and save *X_train, Y_train, X_test* numpy files to the local directory. This will be easier to load for running files that train and test different models, to ensure that preprocessing does not need to occur redundantly.

# Naive Model
*SceneClassification.ipynb* contains the original code to train the Naive model. 

*naive_model.h5* is the saved model after training. With this file, you do not need to rerun SceneClassification.ipynb

*naive_model_train_history.pickle* holds the train accuracy and validation accuracy for the naive model during training. This will be used to plot the statistics of the model.

# Tuned Model
*TunedCNNTrain.py* contains the code used to build and train our second model.

*tuned_model.keras* is the saved model after training. With this file, you do not need to rerun the training code.

*tuned_model_train_history.npy* holds the train accuracy and validation accuracy for the model during training. This will be used to plot the statistics of the model.

### Making the Tuned Model Overfit
We used *TunedCNNTrain.py* and increased the training epochs to 300 to make this model overfit. 

*tuned_model_overfit.keras* is the saved model after training. With this file, you do not need to rerun the training code.

*tuned_model_overfit_train_history.npy* holds the train accuracy and validation accuracy for the model during training. This will be used to plot the statistics of the model.

# Regularized Custom Model
*RegularizedModelTrain.py* contains the code used to build and train our custom model.

*regularized_model.keras* is the saved model after training. With this file, you do not need to rerun the training code.

*regularized_train_history.npy* holds the train accuracy and validation accuracy for the model during training. This will be used to plot the statistics of the model.

# Transfer Learning Model
*TransferLearning.ipynb* contains the code to build and train our transfer learning model. We used the VGG16 model trained with ImageNet as the convolutional base and then added two Dense layers to the architecture.

*transfer_learning_history.npy* holds the train accuracy and validation accuracy for the model during training. 

***BECAUSE OF VGG16 SIZE AND COMPLEXITY, github would not allow it to be posted to this repo. Feel free to run TransferLearning.ipynb to get the model saved to your machine.

# Testing the Models
Use *TunedCNNTest.py* to test any of the models on an image from the test dataset. 

Use *TestGoogleImage.py* to test any of the models on an external image. All you need is the url of the image you want to test.

# Visualizing the Models
*ModelStats.ipynb* contains all graphs of training and validation performance of all the models.

*VisualizeModels.ipynb* shows the intermediate activations for the regularized model, but can be updated to show for any model. 



