FROM yazansh/torch-lightning:latest


COPY requirements.txt additional-requirements.txt
RUN pip install -r additional-requirements.txt

COPY kaggle.json /root/.kaggle/


