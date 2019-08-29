from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import os
import argparse

#Handle Image
def extract_color_stats(image, featureType):
    # split the input image into its respective RGB color channels
    # and then create a feature vector with 6 values : the mean and
    # standard deviation for each of the 3 channels, respectively
    (R, G, B) = image.split()
    if featureType == 2:
        features = [np.min(R), np.min(G), np.min(B),
                    np.mean(R), np.mean(G), np.mean(B),
                    np.max(R), np.max(G), np.max(B),
                    np.std(R), np.std(G), np.std(B)]
    elif featureType == 3:
        features = [np.median(R), np.median(G), np.median(B),
                    np.mean(R), np.mean(G), np.mean(B),
                    np.std(R), np.std(G), np.std(B)]
    else :
        features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
    #Return our set of features
    return features

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
    help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
    help="type of python machine learning model to use")
ap.add_argument("-t", "--testing", type=str, default="",
    help="get sample image that want to be tested")
ap.add_argument("-f", "--featureType", type=int, default=1,
    help="Memilih cara mengekstrak features")
args = vars(ap.parse_args())

#define the dictionary of models our script can use, where the key
#to the dictionary is the name of the model (supplied via comman
#line argunment) and the value is the model itself
models = {
    "knn" : KNeighborsClassifier(n_neighbors = 1),
    "naive_bayes" : GaussianNB(),
    "logit" : LogisticRegression(solver="lbfgs", multi_class = "auto"),
    "svm" : SVC(kernel = "linear", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators = 100),
    "mlp" : MLPClassifier(max_iter = 1000)
}

print("featureType : stat",args["featureType"])
#Grab Images
#grab all image paths in the input dataset directory, initialize our
#list of extracted features and corresponding labels
print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"])#Membaca gambar
data = []
labels = []

#loop over out input Images
for imagePath in imagePaths:
    # load the input image from disk, compute color channel
    # statistics, and then update our data list
    image = Image.open(imagePath)
    features = extract_color_stats(image, args["featureType"])
    data.append(features)
    # extract the class label from the file path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]#diextract dari nama foldernya
    labels.append(label)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# perform a training
# using 75% of the data for training and 25% for evaluation

print("[INFO] loading data...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, random_state=3, test_size = 0.2)

# train the model
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# make predictions on our data and show a classification classification_report
print("[INFO] evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names = le.classes_))


if args['testing'] != "" :
    imageTest = Image.open(args["testing"])
    #featureTestX = np.array(extract_color_stats(imageTest)).reshape(2,3)
    featureTestX = extract_color_stats(imageTest, args["featureType"])
    predictTest = model.predict([featureTestX])

    print("Prediction: ", le.classes_[predictTest])
