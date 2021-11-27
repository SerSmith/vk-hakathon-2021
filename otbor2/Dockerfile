FROM python:3.9.4
RUN mkdir ./otbor2
COPY ./requirements.txt ./otbor2
RUN pip3 install -r ./otbor2/requirements.txt
COPY ./algorithm_data ./otbor2/algorithm_data
COPY ./src ./otbor2/src
COPY ./simulated_test_data /tmp/data
CMD python3 ./otbor2/src/predict.py