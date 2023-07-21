import re
import pandas as pd
from dateutil import parser
from datetime import datetime
from dateutil.parser import parse

df = pd.read_excel(r"C:\Users\ZA40142720\Downloads\validation_Spacy_trf_19072023_1147.xlsx", engine = 'openpyxl')
dff = df[['DATE_RANGE', 'DATE']][df['DATE'].notnull()]
def join_text_by_index(str_date):
    split_val = re.split('(\d+)', str_date)
    index_list = []
    parse_result_list = []
    for i, j in enumerate(split_val):
        if j.isdigit():
            parse_result_list.append(True)
            index_list.append(i)
    print(parse_result_list)
    print(index_list)
    if len(index_list)>1 and len(parse_result_list)>1:
        start_index = index_list[0]
        end_index = index_list[-1] + 1
        joined_text = ''.join(split_val[start_index:end_index])
        return joined_text
    else:
        return "NaN"
    
def clean_and_parse_date(date_str):
    cleaned_dates = []
    if isinstance(date_str, float) and np.isnan(date_str):
        return "NaN"
    else:
        prefixes = ['on', 'dated', 'date', 'dtd','date:']
        cleaned_str = str(date_str).replace('\n', '').strip().lower()
        for prefix in prefixes:
            if cleaned_str.startswith(prefix):
                cleaned_str = cleaned_str[len(prefix):].strip()
                if len(cleaned_str)==0:
                    print("HHHHHIIII")
        #         if isinstance(date_str, float) and np.isnan(date_str):
        #             return "NaN"
        # return "NaN"
        try:
            parsed_date = parser.parse(cleaned_str, fuzzy=True, dayfirst = True, default = None, ignoretz = True)
            formatted_date = parsed_date.strftime('%Y-%m-%d')   #format (e.g., 'YYYY-MM-DD')
            cleaned_dates.append(formatted_date)
    
        except Exception as e:
            cleaned_str_2 = join_text_by_index(cleaned_str)
            if cleaned_str_2:
                parsed_date = parser.parse(cleaned_str_2, fuzzy=True, dayfirst = True, default = None, ignoretz = True)
                formatted_date = parsed_date.strftime('%Y-%m-%d')   #format (e.g., 'YYYY-MM-DD')
                cleaned_dates.append(formatted_date)
                return cleaned_dates
            else:
                return "NaN"
    
        
dff['CLEANED_DATE'] =dff['DATE'].apply(clean_and_parse_date)
dff