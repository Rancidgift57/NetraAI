# NetraAI

NetraAI is an advanced, open-source assistive technology platform designed to enhance accessibility for visually impaired individuals. By integrating state-of-the-art computer vision, natural language processing (NLP), and text-to-speech (TTS) technologies, NetraAI provides real-time audio descriptions of visual environments, enabling users to navigate, interact, and understand their surroundings with greater independence. The project is built with modularity, scalability, and inclusivity in mind, supporting multiple platforms and languages to cater to a global audience.

## Table of Contents
- [Project Vision](#project-vision)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Project Vision
NetraAI aims to bridge the accessibility gap for visually impaired individuals by transforming visual information into detailed, natural-sounding audio descriptions. The project leverages cutting-edge AI to provide an intuitive, user-friendly experience that empowers users to engage with their environment confidently. Our goal is to create a robust, open-source tool that can be extended by the community to support diverse use cases and accessibility needs worldwide.

## Features
- **Real-Time Visual Analysis**:
  - Detects objects, people, faces, and scenes in real-time using a live camera feed.
  - Recognizes text in images (e.g., signs, labels) with Optical Character Recognition (OCR).
  - Identifies colors, spatial relationships, and environmental context.
- **Natural Audio Descriptions**:
  - Converts visual data into coherent, context-aware audio descriptions.
  - Supports customizable description verbosity (brief, standard, detailed).
  - Uses natural-sounding TTS voices for an engaging user experience.
- **Offline Mode**:
  - Enables core functionality (object detection, basic descriptions) without an internet connection using on-device models.
- **Customizable Settings**:
  - Adjust description frequency, voice pitch, speed, and language.
  - Configure camera settings (e.g., resolution, frame rate) for optimal performance.
- **Extensible Architecture**:
  - Modular design allows developers to integrate new models, features, or hardware.
  - Supports custom plugins for specialized use cases (e.g., indoor navigation, facial recognition).
- **Accessibility-First Design**:
  - Built with input from the visually impaired community to ensure usability.
  - Includes voice command support for hands-free operation (beta).

## System Requirements
- **Hardware**:
  - Device with a camera (mobile phone, webcam, or wearable device).
  - Minimum 4GB RAM for smooth performance.
  - Optional: GPU (NVIDIA CUDA-compatible) for faster model inference.
- **Software**:
  - Python 3.8 or higher.
  - Operating Systems: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+), or mobile OS (iOS 14+, Android 10+).
  - Internet connection for initial setup and online mode (optional for offline use).
- **Dependencies**:
  - Listed in `requirements.txt` (e.g., OpenCV, PyTorch, gTTS).
  - Pre-trained models for computer vision and NLP (downloaded separately).

## Installation
Follow these steps to set up NetraAI on your local machine or device:

### Prerequisites
- Install Python 3.8+ from [python.org](https://www.python.org/downloads/).
- Install `git` for cloning the repository.
- Ensure `pip` is updated: `pip install --upgrade pip`.
- For GPU support, install CUDA and cuDNN (refer to [NVIDIA's documentation](https://developer.nvidia.com/cuda-downloads)).

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/netraai/netraai.git
   cd netraai
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**:
   - Visit the model repository links provided in `MiDaS Models`.

5. **Run NetraAI**:
   ```bash
   python NetraAI2.0.py
   ```

## Usage
NetraAI can be used in various scenarios to assist visually impaired users. Below are the steps to get started and example use cases:

### Running the Application
1. **Start NetraAI**:
   ```bash
   python src/NetraAI2.0.py 
   ```
   - Use `--mode offline` for offline functionality (limited features).
   - Use `--device <device_id>` to specify a camera (e.g., `0` for default webcam).

2. **Interact with the System**:
   - Point the camera at your surroundings to receive audio descriptions.
   - Use voice commands (if enabled) to adjust settings or request specific information (e.g., "Read text" or "Describe scene").
   - Adjust settings via the configuration file or a GUI (if implemented).

3. **Example Use Cases**:
   - **Street Navigation**:
     - Point the camera at a street to hear: "A pedestrian crossing with a green light, 10 meters ahead. A bicycle is approaching from the left."

4. **Command-Line Options**:
   ```bash
   python src/NetraAI2.0.py.py --help
   ```
   - `--config <path>`: Specify the configuration file.
   - `--mode <online|offline>`: Set processing mode.
   - `--verbose`: Enable detailed logging for debugging.

## Configuration
The `config/config.yaml` file allows customization of NetraAI's behavior. Key settings include:
- **Camera**:
  - `device_id`: Camera index (e.g., `0` for default).
  - `resolution`: Camera resolution (e.g., `1280x720`).
  - `fps`: Frames per second (e.g., `30`).
- **TTS**:
  - `voice`: TTS voice (e.g., `en-us`, `es-es`).
  - `speed`: Speech speed (e.g., `1.0` for normal).



## Technologies Used
- **Computer Vision**:
  - **OpenCV**: Image processing and camera handling.
  - **YOLOv5**: Real-time object detection.
  - **Tesseract**: Optical Character Recognition for text extraction.
- **Natural Language Processing**:
  - **Hugging Face Transformers**: Text generation for natural descriptions.
  - **spaCy**: Entity recognition and sentence parsing.
- **Text-to-Speech**:
  - **gTTS**: Google Text-to-Speech for online mode.
  - **PyTTSX3**: Offline TTS engine.
- **Core Frameworks**:
  - **Python**: Primary programming language.
  - **PyTorch**: Machine learning framework for vision and NLP models.
  - **TensorFlow**: Optional for custom or alternative models.


## Contributing
We welcome contributions from developers, accessibility advocates, and the community. To contribute:

1. **Fork the Repository**:
   - Create a fork on GitHub: [https://github.com/Rancidgift57/NetraAI.git](https://github.com/Rancidgift57/NetraAI.git).

2. **Set Up Your Environment**:
   - Follow the [Installation](#installation) steps.

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Commit Changes**:
   ```bash
   git commit -m "Add your feature description"
   ```

6. **Push and Open a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   - Submit a pull request with a detailed description and reference any related issues.

7. **Code Review**:
   - Respond to feedback from maintainers and update your pull request as needed.


## License
NetraAI is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
- **Email**: nnair7598@gmail.com
- **GitHub**: [https://github.com/Rancidgift57](https://github.com/Rancidgift57)
- **Twitter/X**: Follow us at [@nnair7083](https://x.com/nnair7083?t=gVaNfC2s0Iek748pqJN_zw&s=09)

## OUTPUT: https://youtu.be/XLRd4JxEQOM


Thank you for supporting NetraAI! Let's make the world more accessible together.
