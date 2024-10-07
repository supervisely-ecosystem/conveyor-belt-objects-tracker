FROM supervisely/base-py-sdk:6.73.192

RUN pip3 install scikit-learn==1.3.2
RUN pip3 install git+https://github.com/supervisely-ecosystem/LightGlue.git@main

LABEL python_sdk_version=6.73.192