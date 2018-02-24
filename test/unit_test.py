import unittest
import pandas as pd
from codes.config import config

class UnitTest(unittest.TestCase):
    time_series1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    time_series2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10

    def test_market_pass_days(self):
        from codes.market import market
        date1 = '2017-01-03'
        self.assertEqual(str(market.pass_days(date1, 1).date()), '2017-01-04')
        self.assertEqual(str(market.pass_days(date1, 6).date()), '2017-01-11')

        config.start_date = pd.to_datetime('2017-01-03')
        market._pass_a_week()
        self.assertEqual(str(config.start_date), '2017-01-10')

    def test_get_historical_data(self):
        config.start_date = pd.to_datetime('2015-11-20')

        from codes.market import market
        pattern = market.get_data(start_date=config.start_date, code='603883.SH')
        self.assertEqual(pattern.shape[0], 30)
        self.assertEqual(str(market.current_date.date()), '2015-12-31')

    def test_pred_ratio(self):
        print()

if __name__ == '__main__':
    unittest.main()