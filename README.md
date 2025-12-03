# Sign Language Detection Portal

This project implements a Sign Language Detection Portal with a Graphical User Interface (GUI) built using Python and Tkinter. It allows users to detect sign language gestures from either a live camera feed or by browsing image files. The system utilizes image processing techniques from OpenCV and a machine learning model (SGDClassifier) from scikit-learn for classification.

## Features

-   **Interactive GUI:** User-friendly interface for easy interaction.
-   **Camera Input:** Real-time sign language detection using your webcam.
-   **Image Browsing:** Detect signs from static image files.
-   **Model Training & Persistence:** Automatically trains a classification model if one is not found, and saves/loads trained models using `joblib`.
-   **Preprocessing Visualization:** Displays intermediate image processing steps (Grayscale, Threshold, Contours) for better understanding.
-   **Confidence Score:** Shows the confidence level of the predicted sign.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/gesture_recognition.git
    cd gesture_recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your dataset:**
    The model requires a dataset of sign language images organized into subdirectories, where each subdirectory represents a specific sign. For example:

    ```
    dataset/
    ├── A/
    │   ├── 3001.jpg
    │   ├── 3002.jpg
    │   └── ...
    ├── B/
    │   ├── 3001.jpg
    │   ├── 3002.jpg
    │   └── ...
    └── Nothing/
        ├── 3001.jpg
        ├── 3002.jpg
        └── ...
    ```
    Supported image formats include `.jpg`, `.jpeg`, `.png`, and `.bmp`.

2.  **Run the application:**
    ```bash
    python sign_language_detector.py
    ```

3.  **Using the GUI:**
    -   **"Use Camera"**: Starts your webcam feed for real-time detection. A green box will appear on the feed; place your hand within this box for detection.
    -   **"Browse Image"**: Opens a file dialog to select an image from your computer for sign detection.
    -   **"Reset Model"**: Deletes the existing trained model and label encoder, forcing the application to retrain the model upon next startup or action that requires the model.
    -   **"Exit"**: Closes the application.

## Model Training

Upon the first run or after resetting the model, the application will automatically train the classification model using the images found in the `dataset` directory. This process might take some time depending on the size of your dataset. A progress window will be displayed during training.

## Project Structure

```
gesture_recognition/
├── README.md
├── requirements.txt
├── sign_language_detector.py
├── dataset/
│   ├── A/
│   ├── B/
│   └── ... (contains subdirectories for each sign, e.g., C, D, E, ..., Z, Nothing, Space)
│       └── 3001.jpg (image files for each sign)
└── models/
    ├── label_encoder.joblib
    └── sign_language_model.joblib
```

## Dependencies

The project relies on the following Python libraries:

-   `opencv-python`
-   `numpy`
-   `matplotlib`
-   `gTTS`
-   `playsound`
-   `pygame`
-   `scikit-learn`
-   `Pillow`
-   `joblib`
-   `tqdm`

These are listed in `requirements.txt` and can be installed using `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
