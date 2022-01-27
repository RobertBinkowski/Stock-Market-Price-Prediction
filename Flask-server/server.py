from flask import Flask, redirect, url_for, render_template, request
from algorithms import *
import pickle

app = Flask(__name__)
home_page = "index.html"


@app.route("/home")
@app.route("/")
def home():
    return render_template(home_page)


@app.route("/home")
@app.route("/", methods=["POST", "GET"])
def data():
    stock = ""
    stock_tag = request.form["stock_tag"]
    stock = round(float(getPrice(stock_tag)), 2)
    chart = getChart()
    return render_template(home_page, stock=stock, chart=chart)


@app.route("/<stock>")
def prediction(stock):
    stock = round(float(getPrice(stock)), 2)
    return render_template(home_page, stock=stock)


if __name__ == "__main__":
    app.run(debug=True)
