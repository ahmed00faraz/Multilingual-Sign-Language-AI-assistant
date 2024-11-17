# Multilingual Sign Language AI Assistant for Airports

This project aims to improve airport accessibility for sign language users by providing a real-time AI assistant capable of understanding and responding to sign language gestures. The system utilizes computer vision, machine learning, and a large language model (LLM) to facilitate communication between travelers and airport staff.

## Features

* **Real-time Sign Language Recognition:** Uses MediaPipe and a custom-trained machine learning model to translate sign language gestures into text.
* **LLM Integration:** Integrates with Gemini to interpret user intent and generate natural language responses.
* **Contextual Understanding:** Trained to understand airport-specific terminology and provide relevant information.
* **User-Friendly Interface:** Provides visual feedback of recognized gestures and displays assistant responses.
* **Multilingual Support:** Leverages the LLM's multilingual capabilities to support diverse travelers.

## How it Works

1. **Hand Tracking & Landmark Detection:** MediaPipe captures hand movements and extracts key landmarks.
2. **Gesture Classification:** The machine learning model classifies the hand landmarks into specific words.
3. **Sentence Formation:** Recognized words are combined to form sentences.
4. **LLM Interaction:** The formed sentence is passed to Gemini, which interprets the user's intent and generates a response.
5. **Output:** The assistant's response is displayed on the screen and spoken aloud using text-to-speech.

## Installation

1. Clone the repository: `git clone https://github.com/ahmed00faraz/Multilingual-Sign-Language-AI-assistant.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your Gemini API key in a `.env` file.

## Usage

1. Run the main application: `python inference_classifier.py`
2. Use sign language to interact with the assistant.

## Data Collection & Training

The `collect_imgs.py` and `create_dataset.py` scripts are used for collecting image data and creating the training dataset. The `train_classifier.py` script trains the machine learning model.

## Evaluation

The `evaluate_model.py` script evaluates the performance of the trained model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
