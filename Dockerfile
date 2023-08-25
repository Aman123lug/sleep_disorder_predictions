FROM python:3.10-slim

WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

EXPOSE 5050

CMD streamlit run webapp.py

