import yfinance as yf


class StockDataSpider:

    @classmethod
    def crawl(cls, code):
        code = StockDataSpider.__transform_code(code)
        ticker = yf.Ticker(code)
        df = ticker.history(period='max')
        df = df.reset_index()
        df['Date'] = df['Date'].apply(lambda x: x.date())
        return df

    @staticmethod
    def __transform_code(code):
        if code[0] == '6':
            return code + ".SS"
        elif code[0] == '3' or code[0] == '0':
            return code + ".SZ"
        else:
            raise ValueError('{0} is an invalid code'.format(code))