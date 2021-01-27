 # 0x06. Keras
 Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
 Keras is the most used deep learning framework among top-5 winning teams on Kaggle. Because Keras makes it easier to run new experiments, it empowers you to try more ideas than your competition, faster.


## 0. Sequential
```sh
model = k.Sequential()
 model.add(k.layers.Dense()
 model.add(k.layers.Dropout(1-keep_prob))
```
## 1. Input
```sh
k.Input(shape=(nx,))
```
## 2. Optimize
```sh
k.optimizers.Adam()
```

## 3. One Hot
```sh
k.utils.to_categorical()
```
## 4. Train 
```sh
network.fit()
```
## 5. Validate
## 6. Early Stopping
## 7. Learning Rate Decay
## 8. Save Only the Best
## 9. Save and Load Model
```sh
 network.save(filename)
 network = K.models.load_model(filename)
 ```
 ## 10. Save and Load Weights
 ```sh
 network.save_weights(filename, save_format=save_format)
 network.load_weights(filename)
 ```
 ## 11. Save and Load Configuration
 saves a model’s configuration in JSON format
 ## 12. Test
 ```sh
 network.evaluate(x=data, y=labels, verbose=verbose)
 ```
## 13. Predict
```sh
network.predict(data, verbose=verbose)
```