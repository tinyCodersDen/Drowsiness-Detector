# This is the code for creating a tensorflow model
# This is currently in-progress, however it gives you around 66% accuracy
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("New_Drowsy.csv")
l = df.loc[:,df.columns!='Status']
y= df['Status'].tolist()
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(51, activation='relu', input_dim=(102), kernel_initializer='normal'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(17, activation='relu', kernel_initializer='normal'),
  tf.keras.layers.Activation('relu'),
  tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='normal'),
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
y = pd.DataFrame({'Status':y})
print(l.shape,y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(l,y,test_size = 0.2, shuffle = True, random_state=2)
model.fit(l,y, epochs=35, batch_size = 1)
class_names = ['Not Drowsy','Drowsy']
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print('\nTest accuracy:', test_acc)
model.save('Drowsy_model.h5')
