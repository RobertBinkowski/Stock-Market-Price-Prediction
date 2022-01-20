from flask import Flask, redirect, url_for, render_template
from algorithms import *
import pickle

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def data():
    stock_tag = 'AAPL'
    data = getPrice(stock_tag)
    return render_template("index", data=data)


if __name__ == "__main__":
    app.run(debug=True)
