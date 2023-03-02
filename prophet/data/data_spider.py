import yfinance as yf


class StockDataSpider:

    @classmethod
    def crawl(cls, symbol):
        symbol = StockDataSpider.__transform_symbol(symbol)
        ticker = yf.Ticker(symbol)
        history = ticker.history(period='max')
        history = history.reset_index()
        history['Date'] = history['Date'].apply(lambda x: x.date())
        return history

    @staticmethod
    def __transform_symbol(symbol):
        if symbol[0] == '6':
            return symbol + ".SS"
        elif symbol[0] == '3' or symbol[0] == '0':
            return symbol + ".SZ"
        else:
            raise ValueError('{0} is an invalid symbol'.format(symbol))
