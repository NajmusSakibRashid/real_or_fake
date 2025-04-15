# Requirements
- GPU (This code is not device agnostic so GPU is mandatory)
- at least Python 3.12.3
- at least pip 24.0.0
# Steps to run the code:
- Clone the directory
- Keep the weight files in the same directory
- Create a virtual environment
  ```sh
  python3 -m venv ./venv
  ```
- Install the packages
  ```sh
  pip install -r requirements.txt
  ```
- Download the input image with pth extension
- Run the code
  ```sh
  python3 main.py <replace with the input image directory>
  ```
# Output
- You will get generated_image.png and generated_<base name of the input file>.pth
  - generated_image.png will be a front, side and top view of the generated file
  - generated_<base name of the input file>.pth will be the generated image file
