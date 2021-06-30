#1. Load data va chia Train, Val va Test
import numpy
from numpy import  loadtxt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model

dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

X = dataset[:, 0:8]
Y = dataset[:, 8]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2)

# model = Sequential()
# model.add(Dense(16, input_dim =8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, Y_train, epochs=100, batch_size=8, validation_data=(X_val, Y_val))
#
# model.save("mymodel.h5")

model = load_model("mymodel.h5")

loss, acc = model.evaluate(X_test, Y_test)
print("Loss = ", loss)
print("Acc = ", acc)

X_new = X_test[10]
Y_new = Y_test[10]

X_new = numpy.expand_dims(X_new, axis=0)

Y_predict = model.predict(X_new)
print("Gia tri du doan = ", Y_predict)
print("Gia tri dung la = ", Y_new)