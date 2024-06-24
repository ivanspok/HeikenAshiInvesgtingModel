import time
from datetime import datetime

# schedule_usa ={'start_hour' : 16, 'start_minute' : 30, 'end_hour' : 20}

def f_range(value, low, high):
  if value >= low and value<= high: return True
  else: return False


def in_shedule(start_week_day = 0, end_week_day = 4 , start_hour = 7 , end_hour = 23, start_minute = 0, end_minute = 60):
    
    current_week_day = datetime.now().weekday()
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    current_second = datetime.now().second

    # # simulation
    # current_week_day = 4
    # current_hour = 0
    # current_minute = 48
    # current_second = 0
    
    # if f_range(current_week_day, start_week_day, end_week_day) and f_range(current_hour, start_hour, end_hour) \
    #      or current_hour == end_hour and current_minute <= end_minute:
    if f_range(current_week_day, start_week_day, end_week_day) \
        and ( (current_hour == start_hour and current_minute >= start_minute) \
            or (f_range(current_hour, start_hour + 1, end_hour -1) and end_hour >= start_hour + 2) \
            or (current_hour == end_hour and current_minute < end_minute) ):
        return True
    else:
        return False


def market_shedule():

    return in_shedule(start_hour = 7, end_hour = 23) or  \
        in_shedule(start_week_day = 0, end_week_day = 5, start_hour = 0, end_hour = 1, end_minute = 30)

def buy_for_long_shedule():
    return in_shedule(start_hour = 7, start_minute = 20, end_hour = 9, end_minute = 55) \
        or in_shedule(start_hour = 10, start_minute = 20, end_hour = 16,  end_minute = 25) \
        or in_shedule(start_hour = 16, start_minute = 45, end_hour = 23)  \
        or in_shedule(start_week_day = 0, end_week_day = 5, start_hour = 0, end_hour = 1, end_minute = 30)

def rus_market_before_usa_shedule():

     return in_shedule(start_hour = 7, end_hour = 15)


def usa_market_shedule():

     return in_shedule(start_hour = 16, end_hour = 23) or in_shedule(start_hour = 0, end_hour = 1, end_minute = 30)

def short_shedule():

    return in_shedule(start_hour = 7, start_minute = 15, end_hour = 9, end_minute = 55) \
        or in_shedule(start_hour = 10, start_minute = 15, end_hour = 16,  end_minute = 20) \
        or in_shedule(start_hour = 16, start_minute = 45, end_hour = 22, end_minute = 50) \

def float_short_long_shedule():
    
    return in_shedule(start_week_day = 0, end_week_day = 4, start_hour = 19, start_minute = 0, end_hour = 21, end_minute = 30) \
        or in_shedule(start_week_day = 5, end_week_day = 5, start_hour = 19, start_minute = 0, end_hour = 20, end_minute = 30)

def short_long_shedule():

    return in_shedule(start_week_day = 0, end_week_day = 4, start_hour = 19, start_minute = 0, end_hour = 21, end_minute = 30) \
        or in_shedule(start_week_day = 5, end_week_day = 5, start_hour = 19, start_minute = 0, end_hour = 21, end_minute = 00)
            # in_shedule(start_hour = 11, start_minute = 15, end_hour = 15,  end_minute = 5) \
            # in_shedule(start_hour = 7, start_minute = 15, end_hour = 9, end_minute = 55) \
        
# print(f'market shedule is {market_shedule()}')
# print(f'short shedule is {short_shedule()}')
# print(f'buy_for_long shedule is {buy_for_long_shedule()}')