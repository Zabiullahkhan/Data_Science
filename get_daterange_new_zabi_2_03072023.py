import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from word2number import w2n

def extract_and_parse_months(strings):
    correct_month_mapping = {
        "jan": "january", "feb": "february", "mar": "march", "apr": "april", "may": "may", "jun": "june",
        "jul": "july", "aug": "august", "sep": "september", "oct": "october", "nov": "november", "dec": "december"
    }
    month_names = []
    pattern = r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"

    #for string in strings:
    matches = re.findall(pattern, strings.lower())
    month_names.extend(matches)
        
    if month_names == []:
        return False,False

    corrected_month_names = []
#     parsed_date_ranges = []
    for month in month_names:
        current_year = datetime.now().year
        date_string = f"{month} {current_year}"
        try:
            date = datetime.strptime(date_string, "%B %Y").date()
            start_date = date.replace(day=1) #.strftime('%d-%m-%Y')
            last_day = (date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            end_date = last_day #.strftime('%d-%m-%Y')
#             parsed_date_ranges.append((start_date, end_date))
        except:
            corrected_month = correct_month_mapping[month]
            corrected_month_names.append(corrected_month)
            date_string = f"{corrected_month} {current_year}"
            date = datetime.strptime(date_string, "%B %Y").date()
            start_date = date.replace(day=1) #.strftime('%d-%m-%Y')
            last_day = (date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            end_date = last_day #.strftime('%d-%m-%Y')
#             parsed_date_ranges.append((start_date, end_date))
    return start_date, end_date


def generate_date_range(date_str,num):
    current_date = datetime.now()
    #print("DATE_STR\n",date_str)
    if "month" in date_str:
        
        if "last" in date_str:
            if any(char.isdigit() for char in date_str):
                num_months = int(''.join(filter(str.isdigit, date_str)))
                start_date = current_date - relativedelta(months=num_months)
                end_date = current_date - relativedelta(days=current_date.day)
            elif any(word in date_str for word in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']):
                try:
                    num_months = w2n.word_to_num(date_str)
                    start_date = current_date - relativedelta(months=num_months)
                    end_date = current_date - relativedelta(days=current_date.day)
                except ValueError:
                    start_date = current_date.replace(day=1) - relativedelta(months=1)
                    end_date = current_date - relativedelta(days=current_date.day)
            else:
                start_date, end_date = extract_and_parse_months(date_str)
                if start_date == False and end_date == False:

                    start_date = current_date.replace(day=1) - relativedelta(months=1)
                    end_date = current_date - relativedelta(days=current_date.day)
        else:
            if any(char.isdigit() for char in date_str):
                num_months = int(''.join(filter(str.isdigit, date_str)))
                start_date = current_date - relativedelta(months=num_months)
                end_date = current_date - relativedelta(days=current_date.day)
            elif any(word in date_str for word in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']):
            
                try:
                    num_months = w2n.word_to_num(date_str)
                    start_date = current_date - relativedelta(months=num_months)
                    end_date = current_date - relativedelta(days=current_date.day)
                except ValueError:
                    start_date = current_date.replace(day=1) - relativedelta(months=1)
                    end_date = current_date - relativedelta(days=current_date.day)
            else:
                start_date, end_date = extract_and_parse_months(date_str)
                if start_date == False and end_date == False:
                    start_date = current_date.replace(day=1) - relativedelta(months=1)
                    end_date = current_date - relativedelta(days=current_date.day)

        date_range_start = start_date
        date_range_end = end_date

    elif "week" in date_str:
        if num == False:
            if "last" in date_str:
                date_range_start = today - timedelta(days=today.weekday() + 7)
                date_range_end = date_range_start + timedelta(days=6)
            elif "next" in date_str:
                date_range_start = today + timedelta(days=(7 - today.weekday()))
                date_range_end = date_range_start + timedelta(days=6)

        else:
            date_range_start = today - timedelta(weeks=num)
            date_range_end = today


    elif "day" in date_str:
        if num == False:
            if "last" in date_str:
                date_range_start = today - relativedelta(days=1)
                date_range_end = today #- timedelta(days=today.day)
            elif "next" in date_str:
                date_range_start = today
                date_range_end = date_range_start + relativedelta(days=1)

        else:
            date_range_start = today - timedelta(days=num)
            date_range_end = today
#     elif "years" in date_str:
#         num  = int(date_str.split()[0])
#         date_range_start = today - relativedelta(years = num)
#         date_range_end = date_range_start + relativedelta(years = num)
    elif "year" in date_str:
        print("YEAR Date_STR\n",date_str)
        #for word in date_str:
        #    word = w2n.word_to_num(word)
#        num = int(date_str.split()[0])
        today = datetime.now().date()
        fy_op= ["fy","financialyear","financial year"]
        print("YEARS_NUM \n",num)
        
        if num == False:
            print("Num = False")
            year_list = find_last_years(date_str)#find_last_years function call

            if "last" in year_list:
                #date_range_start = today - relativedelta(years=1)
                start_date = today - relativedelta(years=1)
                #date_range_end = date_range_start + relativedelta(years=1, days=-1)
                end_date = date_range_start + relativedelta(years=1, days=-1)

            elif "next" in year_list:
                #date_range_start = today + relativedelta(years=1)
                start_date = today + relativedelta(years=1)
                #date_range_end = date_range_start + relativedelta(years=1, days=-1)
                end_date = date_range_start + relativedelta(years=1, days=-1)
        elif num != False:
            if "last" in date_str:
                print("date_str ::: ", date_str, type(date_str))
                #r_year = int(part for part in date_str.split() if part.isdigit())
                r_year = int(num)
                print("r_year------->>>", r_year)
                #date_range_start = today - relativedelta(years=r_year)
                start_date = today - relativedelta(years=r_year)
                #date_range_end = date_range_start + relativedelta(years=r_year, days=-1)
                end_date = start_date + relativedelta(years=r_year, days=-1)

        elif any(opt in date_str.lower() for opt in fy_op): #financial year cases
            if "-" not in data_str:
                year_parts = [part for part in date_str.split() if part.isdigit()]
                if len(year_parts) ==2:
                    start_year = int(year_part[0])
                    end_year = int(year_part[1])
                    if start_year <= end_year:
                        #date_range_start = datetime(start_year,4,1).date()
                        start_date = datetime(start_year,4,1).date()
                        #date_range_end = datetime(end_year,3,31).date()
                        end_date = datetime(end_year,3,31).date()
                    else:
                        return None
                else:
                    return None
            elif "-" in data_str:
                year_parts = [part for part in date_str.split('-') if part.isdigit()]
                if len(year_parts) ==1:
                    start_year = int(year_part[0])
                    end_year = start_year + 1
                    start_date = datetime(start_year,4,1).date()
                    end_date = datetime(end_year,3,31).date()
                elif len(year_parts) == 2:
                    start_year = int(year_part[0])
                    end_year = int(year_part[1])
                    if len(year_parts[1])==2:
                        start_date = datetime(start_year,1,1).date()
                        end_date = datetime(end_year ,3,31).date()
                    else:
                        return None
            else:
                return None
        elif "annual statement" in date_str.lower():
            current_year = today.year
            start_date = datetime(current_year,1,1).date()
            end_date = datetime(current_year,12,31).date()
        elif "-" and "annual statement" in date_str:
            year_parts = [part for part in date_str.split('-') if part.isdigit()]
            if (len(year_parts)==1) :
                start_year = int(year_part[0])
                end_year = start_year + 1
                start_date = datetime(start_year,1,1).date()
                end_date = datetime(end_year,12,31).date()
            elif len(year_parts) == 2:
                start_year = int(year_part[0])
                end_year = int(year_part[1])
                if len(year_parts[1])==2:
                    start_date = datetime(start_year,1,1).date()
                    end_date = datetime(end_year + 1,12,31).date()
                else:
                    return None
            else:
                return None
        else: 
            return None
        

        if start_date > today:
            return None
        date_range_start = start_date
        date_range_end = end_date


      #  else:
        #    date_range_start = today
         #   date_range_end = date_range_start + relativedelta(years=1, days=-1)
       
        

    if date_range_start and date_range_end:
        date_range_str = f"{date_range_start.strftime('%d-%m-%Y')} to {date_range_end.strftime('%d-%m-%Y')}"
    else:
        date_range_str = False

    return date_range_str
