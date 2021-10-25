import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("New_Drowsy.csv")
l = df.loc[:,df.columns!='Status']
y= df['Status'].tolist()
model = tf.keras.models.Sequential([
  # tf.keras.layers.SeparableConv1D( 128 , input_shape=(None,None,102) , kernel_size=( 5 ) , strides=1 ),
  # tf.keras.layers.BatchNormalization(),
  # tf.keras.layers.Activation( 'relu' ) ,
  # tf.keras.layers.SeparableConv2D( 128 , kernel_size=( 5 , 5 ) , strides=1 ),
  # tf.keras.layers.BatchNormalization(),
  # tf.keras.layers.Activation( 'relu' ) ,
  # tf.keras.layers.Cropping1D(cropping=1),
  # tf.keras.layers.Conv1D(32, 3, activation='relu',input_shape=(102,102,)),
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