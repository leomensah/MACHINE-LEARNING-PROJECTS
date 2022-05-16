import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

# fig, ax = plt.subplots()

# print(aaron_judge.columns)
# print(aaron_judge.description.unique())
#print(aaron_judge.type.unique())

# aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})
# print(aaron_judge.type.unique())
# print(aaron_judge['plate_x'])
# aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])
# plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c = aaron_judge.type, cmap=plt.cm.coolwarm, alpha=0.5)

###################################################
# BUILDING THE SVM MODEL
####################################################
def svm_model(dataset):
  fig, ax = plt.subplots()
  dataset['type'] = dataset['type'].map({'S':1, 'B':0})
  dataset = dataset.dropna(subset = ['plate_x', 'plate_z', 'type'])

  plt.scatter(dataset.plate_x, dataset.plate_z, c = dataset.type, cmap=plt.cm.coolwarm, alpha=0.5)

  training_set, validation_set = train_test_split(dataset, random_state=1)
  classifier = SVC(kernel='rbf', C=1, gamma = 3)
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  draw_boundary(ax, classifier)
  plt.show()
  print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


svm_model(jose_altuve)
svm_model(aaron_judge)
svm_model(david_ortiz)


