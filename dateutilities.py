import datetime as dt

from datetime import datetime, timedelta
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

class DateUtilities(object):
    def get_trading_close_holidays(self, year):
        inst = USTradingCalendar()
        holiday_datetimes = inst.holidays(dt.datetime(year-1, 12, 31), dt.datetime(year, 12, 31))
        holidays = set()
        for date in holiday_datetimes.date:
            holidays.add(str(date))
        return holidays

    def is_trading_day(self, date):
        if date.weekday() == 5 or date.weekday() == 6:
            # It is weekend
            return False
        holidays = self.get_trading_close_holidays(datetime.today().year)
        if str(date.date()) in holidays:
            return False
        return True

    def get_next_trading_day(self):
        date = datetime.today() + timedelta(days=1)
        if date.weekday() == 5:
            # The next day is saturday, so we skip to the next monday
            date += timedelta(days=2)
        while not self.is_trading_day(date):
            date += timedelta(days=1)
        return str(date.date())

if __name__ == '__main__':
    date_utils = DateUtilities()
    print(date_utils.get_trading_close_holidays(2017))
    print(date_utils.get_next_trading_day())