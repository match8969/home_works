import cv2
import os, glob
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

image_size = (64, 32)
path = os.path.dirname(os.path.abspath(__file__))
path_aragaki = path + '/aragaki_front'
path_noaragaki = path + '/aragaki_front_no'
x = []
y = []


def read_dir(path, label):
    files = glob.glob(path + "/*.jpg")
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, image_size)
        img_data = img.reshape(-1, )
        x.append(img_data)
        y.append(label)


read_dir(path_noaragaki, 0)
read_dir(path_aragaki, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

joblib.dump(clf, 'aragaki_front.pkl')
