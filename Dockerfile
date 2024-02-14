FROM registry.access.redhat.com/ubi9/ubi-minimal

RUN microdnf install -y python3-pip

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

# need Setuptools
# RUN pip3 uninstall -y setuptools 

RUN microdnf remove -y python3-pip

WORKDIR /workspace

COPY llmcord.py .

CMD ["python3", "llmcord.py"]