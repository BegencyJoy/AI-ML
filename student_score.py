import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
import keras
import tensorflow as tf

data=pd.read_csv(r'D:\python2.0\Deep learning\student_exam_scores.csv')

print(data.head())

x=data.drop(['Final_Exam_Score'],axis=1)
y=data['Final_Exam_Score']

print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
Participation_encoder=LabelEncoder()
x['Participation_in_Group_Study']=Participation_encoder.fit_transform(x['Participation_in_Group_Study'])
print(x['Participation_in_Group_Study'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)

model=keras.Sequential([
    keras.layers.Input(shape=(x_train_scaled.shape[1],)),
    keras.layers.Dense(555,activation='relu'),
    keras.layers.Dense(222,activation='relu'),
    keras.layers.Dense(111,activation='relu'),
    keras.layers.Dense(1,activation='linear')
 ])

huber_loss=tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam',loss=huber_loss)

model.fit(x_train_scaled,y_train,epochs=50,batch_size=64,validation_data=(x_test_scaled,y_test))

model_losses=pd.DataFrame(model.history.history)
model_losses.plot()

predictions=model.predict(x_test_scaled)

from sklearn.metrics import mean_absolute_error,mean_squared_error
print(mean_absolute_error(y_test,predictions))
print(mean_squared_error(y_test,predictions))

def student_score():
  print("Enter the student Details:")

  Hours_Studied_Per_Week=input(f"Enter the Hours the student studied per week :")
  Class_Attendance_Percentage=input(f"Enter the Class Attendance Percentage of the student :")
  Assignments_Completed=input(f"Enter the No.of Assignments student completed :")
  Midterm_Exam_Score=input(f"Enter the midterm exam mark of the student :")
  Participation_in_Group_Study=input(f"Enter the participation of student in group study(yes=1,no=0) :")
  Previous_Exam_Average_Score=input(f"Enter the average score in the previous exam :")

  try:
    Hours_Studied_Per_Week=int( Hours_Studied_Per_Week)
    Class_Attendance_Percentage=int(Class_Attendance_Percentage)
    Assignments_Completed=int( Assignments_Completed)
    Midterm_Exam_Score=int(Midterm_Exam_Score)
    Participation_in_Group_Study=int(Participation_in_Group_Study)
    Previous_Exam_Average_Score=int(Previous_Exam_Average_Score)

  except ValueError:
    print("Invalid input.please enter valid input")
    return

  new_data=pd.DataFrame({
    'Hours_Studied_Per_Week':[Hours_Studied_Per_Week],
    'Class_Attendance_Percentage':[Class_Attendance_Percentage],
    'Assignments_Completed':[Assignments_Completed],
    'Midterm_Exam_Score':[Midterm_Exam_Score],
    'Participation_in_Group_Study':[Participation_in_Group_Study],
    'Previous_Exam_Average_Score':[Previous_Exam_Average_Score]
  })


  new_data_scaled=scaler.transform(new_data)


  prediction=model.predict(new_data_scaled)
  print(f"Predicted Total score:{prediction[0][0]:.2f}")

student_score()