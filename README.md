# Mistral Model Deployment and Fine-tuning

## Installation

To set up the required dependencies, follow these steps:

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv mistral_env
   source mistral_env/bin/activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the backend server by executing the following command:
   ```bash
   python backend.py
   ```

2. Launch the frontend server using this command:
   ```bash
   python frontend.py
   ```
   This will generate a public Gradio URL. Open it in a browser to start a chat.

3. Fine-tuning
   ```bash
   python train.py
   ```
   Data will be downloaded to the 'data' folder, and weights will be saved in the 'weights' folder.

