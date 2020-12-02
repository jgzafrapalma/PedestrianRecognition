# PedestrianRecognition
### Description
PedestrianRecognition is a software that obstains and evaluates models that solve the task of recognition of crossing action through self-supervised learning techniques. The self-supervised methods implemented are:
- __Temporal order verification__: verify whether a sequence of input frames is in correct temporal order.

![image](./imgs/Verification.png)  

- __Temporal order recognition__: recognize the order of a sequence of input frames.

![image](./imgs/OrderPrediction.png)  


# Required software and libraries

- Ubuntu 18.04 OR 16.04 LTS
- Python 3.6
- CUDA 10.1
- cuDNN 7.6
- TensorRT 6.0
- Tensorflow 2.2.0
- Keras 2.3.0

To install the rest of the libraries, follow these steps:

```bash
# Repository files
$ git clone git@github.com:YorYYi/PedestrianRecognition.git
$ cd PedestrianRecognition/

#Create virtual environment
$ python -m venv virtual_environmment_path/virtual_environmment_name

#Activate virtual environment
$ source virtual_environmment_path/virtual_environmment_name

# Install requirements
$ pip install -r requirements.txt
```

# Download dataset




# Create instances

To create the instances with which the models are trained, the following repository scripts must be executed.


```bash
$ python utilities/create_instances_PIE_CrossingDetection.py
```

