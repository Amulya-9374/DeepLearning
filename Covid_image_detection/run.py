# -*- coding: utf-8 -*-
import subprocess
import threading

def run_flask():
    subprocess.run(["python", "flask_covidapp.py"])

def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_covidapp.py"])

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.start()

    flask_thread.join()
    streamlit_thread.join()
