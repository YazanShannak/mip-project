FROM yazansh/pytorch-lightning:latest

RUN apt update && apt install -y zip nano

COPY requirements.txt additional-requirements.txt
RUN pip install -r additional-requirements.txt

COPY kaggle.json /root/.kaggle/


