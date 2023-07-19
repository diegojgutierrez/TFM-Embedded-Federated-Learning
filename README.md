# TFM-Embedded-Federated-Learning

<!-- ABOUT THE PROJECT -->
## About The Project

Implementation of federared learning and differential privacy on an Raspberry Pi and Jetson Nano.

## Built with
- Python
- Tensorflow
- Flower
- Scikit-learn

<!-- USAGE EXAMPLES -->
## Usage

Execute server:
   ```sh
   python server.py
   ```
Execute client/s:
   ```sh
   python client.py True 0 
   ```
   
Execute clients with scripts:
   ```sh
   ./clientScript.sh
   ```

True is the flag for the first iteration of hyperparameter optimization. 0 is the index of the dataset.
