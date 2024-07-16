# Transformer Implementation

This repository contains an implementation of a neural network with dense layers with hyperparameter tuning using math, including ReLU and Softmax activation functions from the paper 'Attention is All you Need, 2017'. The code is written in Python and uses libraries such as NumPy, scikit-learn, and pandas.

## Table of Contents

- [Transformer Implementation](#transformer-implementation)
	- [Table of Contents](#table-of-contents)
	- [Installation](#installation)
	- [Installation](#installation-1)
	- [Usage](#usage)
	- [Progress](#progress)
	- [Hardware Used](#hardware-used)

## Installation

To use this project, you need to have Python installed on your machine. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
	```sh
	git clone <repository-url>
	cd transformer-numpy
	```

2. Create a virtual environment:
	```sh
	python -m venv venv
	```

3. Activate the virtual environment:
	- On Windows:
		```sh
		.\Scripts\activate
		```
	- On Unix or MacOS:
		```sh
		source Scripts/activate
		```

4. Install the dependencies:
	```sh
	pip install -r requirements.txt
	```

## Usage

To train the model, remove the main function and define the network how you want, here an simple example:

    ```python
    from main import DenseLayer
	import numpy as np

	layer = DenseLayer(neurons=10)
	inputs = np.array([[1, 2, 3], [4, 5, 6]])
	relu_output = layer.relu(inputs)
	softmax_output = layer.softmax(inputs)

	print("ReLU Output:", relu_output)
	print("Softmax Output:", softmax_output)
    ```

run the following command:
    ```python
    python run main.py
    ```
## Progress

Future Update
- [X] Transformer 
- [ ] CPU Optimization
- [ ] BERT Implementation
- [ ] CNN
- [ ] Multimodel
- [ ] Web Deployment
  

## Hardware Used

- Processor: AMD Athlon Silver ~2300 Mhz (yeah ik)
- RAM: 12GB DDR4
- GPU: AMD Radeon Integrated Graphic (really bad ik)
- Storage: 1TB SSD
