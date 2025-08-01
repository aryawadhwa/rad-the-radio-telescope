# Radio Telescope Data Project - Setup & Usage Instructions
# =======================================================
#
# 1. Python Environment Setup
# ---------------------------
# - Recommended: Use Python 3.8 or newer (Anaconda or system Python)
#
#   Mac:
#     - Install Homebrew if not present:
#         /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#     - Install Python:
#         brew install python
#     - Or install Anaconda:
#         https://www.anaconda.com/products/distribution
#
#   Windows:
#     - Download Python from:
#         https://www.python.org/downloads/
#     - Or install Anaconda:
#         https://www.anaconda.com/products/distribution
#
# 2. (Optional but recommended) Create a virtual environment:
# ----------------------------------------------------------
#   Mac/Linux:
#     python3 -m venv .venv
#     source .venv/bin/activate
#
#   Windows:
#     python -m venv .venv
#     .venv\Scripts\activate
#
#   Anaconda (Mac/Windows):
#     conda create -n radio_env python=3.10
#     conda activate radio_env
#
# 3. Install required packages:
# -----------------------------
#   Mac/Linux:
#     pip install -r requirements.txt
#
#   Windows:
#     pip install -r requirements.txt
#
#   Anaconda (Mac/Windows):
#     conda install --file requirements.txt
#     # If reportlab fails, run: conda install reportlab
#     # Or: python -m pip install reportlab
#
# 4. Workflow Steps (Terminal Commands):
# --------------------------------------
#   1. Upload and run Arduino sketch:
#        - Open Arduino IDE, load 1_arduino_data_logger.ino, upload to your Arduino.
#   2. Log data from Arduino to CSV:
#        Mac/Linux:
#          python3 2_data_logger.py
#        Windows:
#          python 2_data_logger.py
#   3. (If needed) Extract CSV from RTF/text:
#        Mac/Linux:
#          python3 3_csv_extractor.py
#        Windows:
#          python 3_csv_extractor.py
#   4. Clean your CSV data:
#        Mac/Linux:
#          python3 4_csv_cleaner.py
#        Windows:
#          python 4_csv_cleaner.py
#   5. Analyze and generate reports:
#        Mac/Linux:
#          python3 5_data_analyzer.py
#        Windows:
#          python 5_data_analyzer.py
#
# 5. Troubleshooting:
# -------------------
#   - If you see 'ModuleNotFoundError', ensure you are using the correct Python/conda environment and have installed all requirements.
#   - For serial port issues, check your Arduino port and permissions.
#
# =======================================================

pyserial
matplotlib
pandas
numpy
scipy
seaborn
reportlab 

# Written by Arya Wadhwa 
