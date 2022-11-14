FROM python:3.8-buster
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .
ENTRYPOINT [ "python" ]
EXPOSE 5000
CMD ["home_credit_app.py","--host=0.0.0.0"]