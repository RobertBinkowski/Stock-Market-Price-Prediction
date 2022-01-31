from datetime import date, timedelta

# Main stock class for the algoirithm


class Stock:
    chart = ""
    error_rate = 0

    def __init__(self, tag, pred):
        self.tag = tag
        self.date = date.today().strftime("%Y-%m-%d")
        self.pred = pred

    def get_tag(self):
        return self.tag

    def get_date(self):
        return self.date

    def get_pred(self):
        return self.pred

    def get_chart(self):
        return self.chart

    def set_tag(self, tag):
        self.tag = tag

    def get_error_rate(self):
        return self.error_rate

    def set_error_rate(self, error_rate):
        self.error_rate = round(error_rate)

    def set_pred(self, pred):
        self.date = date.today().strftime("%Y-%m-%d")
        self.pred = pred

    def set_chart(self, chart):
        self.chart = chart

    def to_json(self):  # simple json Format String base for a file
        json_output = {}
        json_output["stockTag"] = str(self.tag)
        json_output["predictionDate"] = date.today().strftime("%Y-%m-%d")
        json_output["prediction"] = str(self.pred)
        json_output["errorRate"] = str(self.error_rate)
        json_output["chart"] = str(self.chart)
        return json_output
