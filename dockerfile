FROM gurobi/python:latest

WORKDIR /app


ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt


COPY . /app

CMD ["python","app.py"]