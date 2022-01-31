from flask import Flask, redirect, url_for, render_template, request
from algorithms import *
from stock import *
import pickle

app = Flask(__name__)
home_page = "index.html"
pred_page = "pred.html"


@app.route("/home")
@app.route("/")
def home():
    return render_template(home_page, stock=0)


@app.route("/home")
@app.route("/", methods=["POST", "GET"])
def data():
    stock = []
    stock_tag = request.form["stock_tag"]
    stock = get_price(stock_tag)
    if stock == 0:
        return render_template(home_page, stock=0)
    return render_template(home_page, stock=stock)


@app.route("/<stock_tag>")
def prediction(stock_tag):
    stock = get_price(stock_tag)
    if stock == 0:
        return render_template(home_page, stock_prediction=0)
    return render_template(home_page, stock=stock)


if __name__ == "__main__":
    app.run(debug=True)
