

"""

Downloading Dataset through Kaggle API.

"""

import os
import sys

cmd = "kaggle datasets download -d muhammadrehan00/chest-xray-dataset -p data/raw/unzipped_raw_data"
exit_code = os.system(cmd)

if exit_code != 0:
    sys.exit(
        "ERROR: Kaggle command failed.\n"
        "Please make sure:\n"
        "- The correct environment is activated\n"
        "- Kaggle is installed (pip install kaggle)\n"
        "- Kaggle authentication is set (KAGGLE_API_TOKEN)"
    )
    
if os.system("unzip -o data/raw/*.zip -d data/interim") != 0:
    sys.exit("ERROR: Unzip failed.")