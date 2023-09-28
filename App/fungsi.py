from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU, GlobalAveragePooling2D

def make_model():
    BaseModel = MobileNet(weights='imagenet', include_top=False, input_shape= (224, 224, 3) )
    model = Sequential()
    model.add(BaseModel)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(6, activation="softmax" , name="classification"))
    
    return model