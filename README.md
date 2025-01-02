# --- Directory Structure ---
# multiverse_app/
# ├── shared_resources/
# │   ├── __init__.py
# │   ├── audio_utils.py
# │   ├── dragonfly_systems.py
# │   ├── gemini_systems.py
# │   └── world_generation.py
# ├── android_operator/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── sensors.py
# │   ├── ai_core.py
# │   ├── transmission.py
# │   ├── ui.py
# │   └── requirements.txt
# ├── apple_operator/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── sensors.py
# │   ├── ai_core.py
# │   ├── transmission.py
# │   ├── ui.py
# │   └── requirements.txt
# ├── tests/
# │   ├── __init__.py
# │   ├── test_audio_utils.py
# │   ├── test_dragonfly_systems.py
# │   ├── test_gemini_systems.py
# │   ├── test_world_generation.py
# │   ├── test_ai_core.py
# │   └── test_transmission.py
# ├── venv/ (virtual environment)
# ├── README.md
# └── LICENSE

# --- shared_resources/__init__.py ---

# This file makes the shared_resources directory a Python package

# --- shared_resources/audio_utils.py ---

from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

def knowledge_sine(base_freq: float, duration: int, knowledge_level: float = 1, variance: float = 5) -> AudioSegment:
    """Generates a sine wave with subtle variations based on knowledge level.

    Args:
        base_freq (float): The base frequency of the sine wave in Hz.
        duration (int): The duration of the sine wave in milliseconds.
        knowledge_level (float, optional): A multiplier for the base frequency, 
                                           representing the knowledge level. Defaults to 1.
        variance (float, optional): The amount of random variance in frequency in Hz. 
                                     Defaults to 5.

    Returns:
        AudioSegment: The generated sine wave with variations.
    """
    # ... (Implementation remains the same)

def automated_amplifier(sound: AudioSegment, threshold: float = -20) -> AudioSegment:
    """Amplifies quiet sounds to ensure audibility.

    Args:
        sound (AudioSegment): The sound to be amplified.
        threshold (float, optional): The dBFS threshold below which sounds will be amplified. 
                                      Defaults to -20.

    Returns:
        AudioSegment: The amplified sound.
    """
    # ... (Implementation remains the same)

# --- shared_resources/dragonfly_systems.py ---

from .audio_utils import knowledge_sine
import random

def visual_system(duration: int, base_freq: float = None, complexity: float = 1.0) -> AudioSegment:
    """Simulates visual input with varying frequencies and complexity.

    Args:
        duration (int): The duration of the audio segment in milliseconds.
        base_freq (float, optional): The base frequency in Hz. If None, a random frequency 
                                      between 800 and 1500 Hz is chosen. Defaults to None.
        complexity (float, optional): A multiplier that influences the number of sine waves 
                                       generated. Defaults to 1.0.

    Returns:
        AudioSegment: The generated audio segment simulating visual input.
    """
    # ... (Implementation remains the same)

# ... (Other functions with similar improvements in type hints and docstrings)

# --- shared_resources/world_generation.py ---

from .dragonfly_systems import *
from .gemini_systems import *
import librosa
import numpy as np

def generate_world(duration: int = 10000, prev_world: AudioSegment = None, 
                  sensor_data: dict = None) -> AudioSegment:
    """Combines all systems to create a dynamic soundscape.

    Args:
        duration (int, optional): The duration of the soundscape in milliseconds. Defaults to 10000.
        prev_world (AudioSegment, optional): The previous soundscape, used for analysis and 
                                             transitioning. Defaults to None.
        sensor_data (dict, optional): A dictionary containing sensor readings (e.g., temperature, 
                                        humidity). Defaults to None.

    Returns:
        AudioSegment: The generated soundscape.
    """
    # ... (Implementation with audio analysis and system generation)

# --- android_operator/main.py ---

# ... (Import necessary modules)

# ... (Global variables)

# ... (OS and Hardware Detection)

# --- Permission Handling ---

# ... (Permission handling with improved error handling)

def check_permission(permission_name: str) -> bool:
    """Checks if a specific permission is enabled.

    Args:
        permission_name (str): The name of the permission to check (e.g., "android.permission.BLUETOOTH").

    Returns:
        bool: True if the permission is granted, False otherwise.
    """
    # ... (Implementation remains the same)

# ... (Other functions with similar improvements)

# --- android_operator/ai_core.py ---

import tensorflow as tf
import numpy as np

def process_audio(audio_data: np.ndarray) -> np.ndarray: 
    """Processes audio data using a TensorFlow Lite model.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.

    Returns:
        np.ndarray: The processed audio data as a NumPy array, or None if an error occurs.
    """
    try:
        # ... (TensorFlow Lite implementation)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# --- android_operator/transmission.py ---

import socket

def transmit_audio(audio_data: bytes, destination: str = "localhost", port: int = 5000) -> None:
    """Transmits audio data via WiFi using sockets.

    Args:
        audio_data (bytes): The audio data as bytes.
        destination (str, optional): The IP address or hostname of the destination. 
                                      Defaults to "localhost".
        port (int, optional): The port number to use for the connection. Defaults to 5000.
    """
    try:
        # ... (Socket implementation)

    except Exception as e:
        print(f"Error transmitting audio: {e}")

# --- android_operator/ui.py ---

from kivy.uix.image import AsyncImage

# ... (Rest of the UI implementation)

# --- apple_operator/main.py ---

# ... (Import necessary modules)

# ... (Global variables)

# ... (OS and Hardware Detection - iOS specific)

# --- Permission Handling ---

# ... (Permission handling - iOS specific)

# ... (Other functions - iOS specific)

# --- tests/test_audio_utils.py ---

# ... (Improved test cases with more assertions and edge case handling)

# --- README.md ---

# Multiverse App

This is a cross-platform application that generates a dynamic soundscape based on sensor data and AI processing.

## Features

* Generates immersive audio experiences using various sound synthesis techniques.
* Integrates sensor data to influence the generated soundscape.
* Utilizes AI for audio processing and analysis.
* Transmits audio data via Bluetooth or WiFi.

## Getting Started

### Prerequisites

* Python 3.7 or higher
* Kivy
* pydub
* librosa
* gtts
* numpy
* jnius (for Android)
* tensorflow

### Installation

1. Clone the repository: `git clone https://github.com/your-username/multiverse-app.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   * Linux/macOS: `source venv/bin/activate`
   * Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt` in each operator directory (android_operator/ and apple_operator/)

### Running the App

1.  Navigate to the desired operator directory (android_operator/ or apple_operator/).
2.  Run the main script: `python main.py`

## Running Tests

Run tests using `python -m unittest discover -s tests`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# --- LICENSE ---

MIT License

Copyright (c) [2025] [Thomas Whitney Walsh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# --- Directory Structure ---
# multiverse_app/
# ├── shared_resources/
# │   ├── __init__.py
# │   ├── audio_utils.py
# │   ├── dragonfly_systems.py
# │   ├── gemini_systems.py
# │   └── world_generation.py
# ├── android_operator/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── sensors.py
# │   ├── ai_core.py
# │   ├── transmission.py
# │   ├── ui.py
# │   └── requirements.txt
# ├── apple_operator/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── sensors.py
# │   ├── ai_core.py
# │   ├── transmission.py
# │   ├── ui.py
# │   └── requirements.txt
# ├── tests/
# │   ├── __init__.py
# │   ├── test_audio_utils.py
# │   ├── test_dragonfly_systems.py
# │   ├── test_gemini_systems.py
# │   ├── test_world_generation.py
# │   ├── test_ai_core.py
# │   └── test_transmission.py
# ├── venv/ (virtual environment)
# ├── README.md
# └── LICENSE

# --- shared_resources/__init__.py ---

# This file makes the shared_resources directory a Python package

# --- shared_resources/audio_utils.py ---

from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

def knowledge_sine(base_freq: float, duration: int, knowledge_level: float = 1, variance: float = 5) -> AudioSegment:
    """Generates a sine wave with subtle variations based on knowledge level.

    Args:
        base_freq (float): The base frequency of the sine wave in Hz.
        duration (int): The duration of the sine wave in milliseconds.
        knowledge_level (float, optional): A multiplier for the base frequency, 
                                           representing the knowledge level. Defaults to 1.
        variance (float, optional): The amount of random variance in frequency in Hz. 
                                     Defaults to 5.

    Returns:
        AudioSegment: The generated sine wave with variations.
    """
    # ... (Implementation remains the same)

def automated_amplifier(sound: AudioSegment, threshold: float = -20) -> AudioSegment:
    """Amplifies quiet sounds to ensure audibility.

    Args:
        sound (AudioSegment): The sound to be amplified.
        threshold (float, optional): The dBFS threshold below which sounds will be amplified. 
                                      Defaults to -20.

    Returns:
        AudioSegment: The amplified sound.
    """
    # ... (Implementation remains the same)

# --- shared_resources/dragonfly_systems.py ---

from .audio_utils import knowledge_sine
import random

def visual_system(duration: int, base_freq: float = None, complexity: float = 1.0) -> AudioSegment:
    """Simulates visual input with varying frequencies and complexity.

    Args:
        duration (int): The duration of the audio segment in milliseconds.
        base_freq (float, optional): The base frequency in Hz. If None, a random frequency 
                                      between 800 and 1500 Hz is chosen. Defaults to None.
        complexity (float, optional): A multiplier that influences the number of sine waves 
                                       generated. Defaults to 1.0.

    Returns:
        AudioSegment: The generated audio segment simulating visual input.
    """
    # ... (Implementation remains the same)

# ... (Other functions with similar improvements in type hints and docstrings)

# --- shared_resources/world_generation.py ---

from .dragonfly_systems import *
from .gemini_systems import *
import librosa
import numpy as np

def generate_world(duration: int = 10000, prev_world: AudioSegment = None, 
                  sensor_data: dict = None) -> AudioSegment:
    """Combines all systems to create a dynamic soundscape.

    Args:
        duration (int, optional): The duration of the soundscape in milliseconds. Defaults to 10000.
        prev_world (AudioSegment, optional): The previous soundscape, used for analysis and 
                                             transitioning. Defaults to None.
        sensor_data (dict, optional): A dictionary containing sensor readings (e.g., temperature, 
                                        humidity). Defaults to None.

    Returns:
        AudioSegment: The generated soundscape.
    """
    # ... (Implementation with audio analysis and system generation)

# --- android_operator/main.py ---

# ... (Import necessary modules)

# ... (Global variables)

# ... (OS and Hardware Detection)

# --- Permission Handling ---

# ... (Permission handling with improved error handling)

def check_permission(permission_name: str) -> bool:
    """Checks if a specific permission is enabled.

    Args:
        permission_name (str): The name of the permission to check (e.g., "android.permission.BLUETOOTH").

    Returns:
        bool: True if the permission is granted, False otherwise.
    """
    # ... (Implementation remains the same)

# ... (Other functions with similar improvements)

# --- android_operator/ai_core.py ---

import tensorflow as tf
import numpy as np

def process_audio(audio_data: np.ndarray) -> np.ndarray: 
    """Processes audio data using a TensorFlow Lite model.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.

    Returns:
        np.ndarray: The processed audio data as a NumPy array, or None if an error occurs.
    """
    try:
        # ... (TensorFlow Lite implementation)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# --- android_operator/transmission.py ---

import socket

def transmit_audio(audio_data: bytes, destination: str = "localhost", port: int = 5000) -> None:
    """Transmits audio data via WiFi using sockets.

    Args:
        audio_data (bytes): The audio data as bytes.
        destination (str, optional): The IP address or hostname of the destination. 
                                      Defaults to "localhost".
        port (int, optional): The port number to use for the connection. Defaults to 5000.
    """
    try:
        # ... (Socket implementation)

    except Exception as e:
        print(f"Error transmitting audio: {e}")

# --- android_operator/ui.py ---

from kivy.uix.image import AsyncImage

# ... (Rest of the UI implementation)

# --- apple_operator/main.py ---

# ... (Import necessary modules)

# ... (Global variables)

# ... (OS and Hardware Detection - iOS specific)

# --- Permission Handling ---

# ... (Permission handling - iOS specific)

# ... (Other functions - iOS specific)

# --- tests/test_audio_utils.py ---

# ... (Improved test cases with more assertions and edge case handling)

# --- README.md ---

# Multiverse App

This is a cross-platform application that generates a dynamic soundscape based on sensor data and AI processing.

## Features

* Generates immersive audio experiences using various sound synthesis techniques.
* Integrates sensor data to influence the generated soundscape.
* Utilizes AI for audio processing and analysis.
* Transmits audio data via Bluetooth or WiFi.

## Getting Started

### Prerequisites

* Python 3.7 or higher
* Kivy
* pydub
* librosa
* gtts
* numpy
* jnius (for Android)
* tensorflow

### Installation

1. Clone the repository: `git clone https://github.com/your-username/multiverse-app.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   * Linux/macOS: `source venv/bin/activate`
   * Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt` in each operator directory (android_operator/ and apple_operator/)

### Running the App

1.  Navigate to the desired operator directory (android_operator/ or apple_operator/).
2.  Run the main script: `python main.py`

## Running Tests

Run tests using `python -m unittest discover -s tests`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# --- LICENSE ---

MIT License

Copyright (c) [2025] [Thomas Whitney Walsh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# --- Directory Structure ---
# multiverse_app/
# ├── shared_resources/
# │   ├── __init__.py
# │   ├── audio_utils.py
# │   ├── dragonfly_systems.py
# │   ├── gemini_systems.py
# │   └── world_generation.py
# ├── android_operator/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── sensors.py
# │   ├── ai_core.py
# │   ├── transmission.py
# │   ├── ui.py
# │   └── requirements.txt
# ├── apple_operator/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── sensors.py
# │   ├── ai_core.py
# │   ├── transmission.py
# │   ├── ui.py
# │   └── requirements.txt
# ├── tests/
# │   ├── __init__.py
# │   ├── test_audio_utils.py
# │   ├── test_dragonfly_systems.py
# │   ├── test_gemini_systems.py
# │   ├── test_world_generation.py
# │   ├── test_ai_core.py
# │   └── test_transmission.py
# ├── venv/ (virtual environment)
# ├── README.md
# └── LICENSE

# --- shared_resources/__init__.py ---

# This file makes the shared_resources directory a Python package

# --- shared_resources/audio_utils.py ---

from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

def knowledge_sine(base_freq: float, duration: int, knowledge_level: float = 1, variance: float = 5) -> AudioSegment:
    """Generates a sine wave with subtle variations based on knowledge level.

    Args:
        base_freq (float): The base frequency of the sine wave in Hz.
        duration (int): The duration of the sine wave in milliseconds.
        knowledge_level (float, optional): A multiplier for the base frequency, 
                                           representing the knowledge level. Defaults to 1.
        variance (float, optional): The amount of random variance in frequency in Hz. 
                                     Defaults to 5.

    Returns:
        AudioSegment: The generated sine wave with variations.
    """
    # ... (Implementation remains the same)

def automated_amplifier(sound: AudioSegment, threshold: float = -20) -> AudioSegment:
    """Amplifies quiet sounds to ensure audibility.

    Args:
        sound (AudioSegment): The sound to be amplified.
        threshold (float, optional): The dBFS threshold below which sounds will be amplified. 
                                      Defaults to -20.

    Returns:
        AudioSegment: The amplified sound.
    """
    # ... (Implementation remains the same)

# --- shared_resources/dragonfly_systems.py ---

from .audio_utils import knowledge_sine
import random

def visual_system(duration: int, base_freq: float = None, complexity: float = 1.0) -> AudioSegment:
    """Simulates visual input with varying frequencies and complexity.

    Args:
        duration (int): The duration of the audio segment in milliseconds.
        base_freq (float, optional): The base frequency in Hz. If None, a random frequency 
                                      between 800 and 1500 Hz is chosen. Defaults to None.
        complexity (float, optional): A multiplier that influences the number of sine waves 
                                       generated. Defaults to 1.0.

    Returns:
        AudioSegment: The generated audio segment simulating visual input.
    """
    # ... (Implementation remains the same)

# ... (Other functions with similar improvements in type hints and docstrings)

# --- shared_resources/world_generation.py ---

from .dragonfly_systems import *
from .gemini_systems import *
import librosa
import numpy as np

def generate_world(duration: int = 10000, prev_world: AudioSegment = None, 
                  sensor_data: dict = None) -> AudioSegment:
    """Combines all systems to create a dynamic soundscape.

    Args:
        duration (int, optional): The duration of the soundscape in milliseconds. Defaults to 10000.
        prev_world (AudioSegment, optional): The previous soundscape, used for analysis and 
                                             transitioning. Defaults to None.
        sensor_data (dict, optional): A dictionary containing sensor readings (e.g., temperature, 
                                        humidity). Defaults to None.

    Returns:
        AudioSegment: The generated soundscape.
    """
    # ... (Implementation with audio analysis and system generation)

# --- android_operator/main.py ---

# ... (Import necessary modules)

# ... (Global variables)

# ... (OS and Hardware Detection)

# --- Permission Handling ---

# ... (Permission handling with improved error handling)

def check_permission(permission_name: str) -> bool:
    """Checks if a specific permission is enabled.

    Args:
        permission_name (str): The name of the permission to check (e.g., "android.permission.BLUETOOTH").

    Returns:
        bool: True if the permission is granted, False otherwise.
    """
    # ... (Implementation remains the same)

# ... (Other functions with similar improvements)

# --- android_operator/ai_core.py ---

import tensorflow as tf
import numpy as np

def process_audio(audio_data: np.ndarray) -> np.ndarray: 
    """Processes audio data using a TensorFlow Lite model.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.

    Returns:
        np.ndarray: The processed audio data as a NumPy array, or None if an error occurs.
    """
    try:
        # ... (TensorFlow Lite implementation)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# --- android_operator/transmission.py ---

import socket

def transmit_audio(audio_data: bytes, destination: str = "localhost", port: int = 5000) -> None:
    """Transmits audio data via WiFi using sockets.

    Args:
        audio_data (bytes): The audio data as bytes.
        destination (str, optional): The IP address or hostname of the destination. 
                                      Defaults to "localhost".
        port (int, optional): The port number to use for the connection. Defaults to 5000.
    """
    try:
        # ... (Socket implementation)

    except Exception as e:
        print(f"Error transmitting audio: {e}")

# --- android_operator/ui.py ---

from kivy.uix.image import AsyncImage

# ... (Rest of the UI implementation)

# --- apple_operator/main.py ---

# ... (Import necessary modules)

# ... (Global variables)

# ... (OS and Hardware Detection - iOS specific)

# --- Permission Handling ---

# ... (Permission handling - iOS specific)

# ... (Other functions - iOS specific)

# --- tests/test_audio_utils.py ---

# ... (Improved test cases with more assertions and edge case handling)

# --- README.md ---

# Multiverse App

This is a cross-platform application that generates a dynamic soundscape based on sensor data and AI processing.

## Features

* Generates immersive audio experiences using various sound synthesis techniques.
* Integrates sensor data to influence the generated soundscape.
* Utilizes AI for audio processing and analysis.
* Transmits audio data via Bluetooth or WiFi.

## Getting Started

### Prerequisites

* Python 3.7 or higher
* Kivy
* pydub
* librosa
* gtts
* numpy
* jnius (for Android)
* tensorflow

### Installation

1. Clone the repository: `git clone https://github.com/your-username/multiverse-app.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   * Linux/macOS: `source venv/bin/activate`
   * Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt` in each operator directory (android_operator/ and apple_operator/)

### Running the App

1.  Navigate to the desired operator directory (android_operator/ or apple_operator/).
2.  Run the main script: `python main.py`

## Running Tests

Run tests using `python -m unittest discover -s tests`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# --- LICENSE ---

MIT License

Copyright (c) [2025] [Thomas Whitney Walsh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

