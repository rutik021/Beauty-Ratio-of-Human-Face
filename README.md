# Beauty Ratio of Human Face

This project aims to analyze the proportions of a human face to determine its adherence to the Golden Ratio, often associated with aesthetic beauty. By utilizing facial landmarks, the program calculates various ratios and compares them to the ideal Golden Ratio.

## Features

- Detect facial landmarks using dlib
- Calculate facial ratios and compare them with the Golden Ratio
- Provide a beauty score based on facial symmetry and proportions

## Requirements

- Python 3.6 or higher
- dlib
- OpenCV
- numpy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Beauty-Ratio-of-Human-Face.git
    cd Beauty-Ratio-of-Human-Face
    ```

2. Create a virtual environment (optional but recommended):
    ```sh
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```



## Usage

1. Ensure you have the necessary model files for dlib. Download the pre-trained shape predictor model from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

2. Run the main script:
    ```sh
    python program.py
    ```


## Acknowledgments

- [dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/)
- The authors of any tutorials, articles, or other resources that helped in the creation of this project.

