# Transformer-From-Scratch

This project is an implementation of a transformer neural network from scratch. It utilizes the IRIS dataset, which consists of flower data. The implementation is similar to other existing implementations but includes more comprehensive code. Included with the database used

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

To train the model, define the network how you want, here an example:
    ```python
    model = Network()
    model.add(DenseLayer(neurons=64))
    model.add(DenseLayer(neurons=32))
    model.add(DenseLayer(neurons=3))
    model.train(X_train, y_train, epochs=1000)
    ```

run the following command:
    ```python
    python run main.py
    ```



