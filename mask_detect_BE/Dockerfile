FROM python:3.6-stretch
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential libgl1-mesa-dev libgl1-mesa-glx
Copy . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5353
CMD [ "python", "./mask-detection-image.py" ]