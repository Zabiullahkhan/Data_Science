from dateutil import parser
import spacy
import pandas as pd
import preprocess as pp
import numpy as np
import re


def convert_amounts(str_amount):
    if pd.isnull(str_amount):
        return "NaN"
    amt_dic = {
    100000: ['lacs', 'lac', 'lakhs', 'lakh', 'lk', 'lkhs'],
    1000: ['k', 'thousand'],
    10000000: ['cr', 'crore', 'crores']
}
    str_amount = str_amount.replace(',', '')
    pattern = r'(\d+(?:\.\d+)?)'
    parts = [part for part in re.split(pattern, str_amount) if part]

    nums = []
    for i in parts:
        i = i.strip()
        if i.isdigit():
            nums.append(int(i))
        else:
            try:
                nums.append(float(i))
            except ValueError:
                for key, values in amt_dic.items():
                    if i in values:
                        if nums[-1] is not None:
                            converted_value = nums[-1] * key
                            nums[-1] = converted_value

    return nums

###-------------- BELOW FUNCTION "clean_and_parse_date" CURRENTLY IS NOT IN USE ------------###
def clean_and_parse_date(date_str):

    cleaned_dates = []
    if isinstance(date_str, float) and np.isnan(date_str):
        #cleaned_dates.append("NaN")  # Return None for NaN values
        return "NaN"
    prefixes = ['on', 'dated', 'date', 'dtd']
    cleaned_str = str(date_str).replace('\n', '').strip().lower()
    for prefix in prefixes:
        if cleaned_str.startswith(prefix):
            cleaned_str = cleaned_str[len(prefix):].strip()
            break
    date_format = '%d/%m'
    try:
        parsed_date = parser.parse(cleaned_str, fuzzy=True, dayfirst = True, default = None, ignoretz = True)
        formatted_date = parsed_date.strftime('%Y-%m-%d')   #format (e.g., 'YYYY-MM-DD')
        cleaned_dates.append(formatted_date)

    except Exception as e:
        # If an exception occurs during parsing, append the cleaned_str to the cleaned_dates list
        cleaned_dates.append(cleaned_str)

    return cleaned_dates


def join_text_by_index(str_date):
    split_val = re.split('(\d+)', str_date)
    index_list = []
    parse_result_list = []
    for i, j in enumerate(split_val):
        if j.isdigit():
            parse_result_list.append(True)
            index_list.append(i)
    if len(index_list)>1 and len(parse_result_list)>1:
        start_index = index_list[0]
        end_index = index_list[-1] + 1
        joined_text = ''.join(split_val[start_index:end_index])
        return joined_text
    else:
        return None


def clean_and_parse_date(date_str):
    cleaned_dates = []
    if isinstance(date_str, float) or pd.isna(date_str):
        return None
    else:
        prefixes = ['on', 'dated', 'date', 'dtd','date:']
        cleaned_str = str(date_str).replace('\n', '').strip().lower()
        for prefix in prefixes:
            if cleaned_str.startswith(prefix):
                cleaned_str = cleaned_str[len(prefix):].strip()
                if len(cleaned_str)==0:
                    print("CAPTURED STRING DOES NOT CONTAINS DATE VALUE")
                    return None

        try:
            parsed_date = parser.parse(cleaned_str, fuzzy=True, dayfirst = True, default = None, ignoretz = True)
            formatted_date = parsed_date.strftime('%Y-%m-%d')   #format (e.g., 'YYYY-MM-DD')
            cleaned_dates.append(formatted_date)
            return cleaned_dates

        except Exception as e:
            cleaned_str_2 = join_text_by_index(cleaned_str)
            if cleaned_str_2:
                try:
                    parsed_date = parser.parse(cleaned_str_2, fuzzy=True, dayfirst = True, default = None, ignoretz = True)
                    formatted_date = parsed_date.strftime('%Y-%m-%d')   #format (e.g., 'YYYY-MM-DD')
                    cleaned_dates.append(formatted_date)
                    return cleaned_dates
                except Exception as e:
                    return [cleaned_str_2]

                    # return cleaned_dates
            else:
                return None


nlp = spacy.load("/home/wipro/NLP_RnD/PB_ENTITY/SpaCy_Trf_17_07_2023/output_17_07_2023/model-best")

dff = pd.read_csv("./validation_statetement_data_1147.csv")

first_column = dff.iloc[:, 0]
first_column = first_column.apply(lambda x: pp.preprocess_text(x))
#spacy.displacy.render(doc, style = 'ent')
data = []
for doc in nlp.pipe(first_column, disable=["tagger", "parser"]):
    entities = {ent.label_: ent.text for ent in doc.ents}
    #data.append({'Statement': pp.preprocess_text(doc.text), 'Entities': entities})
    data.append({'Statement':doc.text, 'Entities': entities})
predicted_df = pd.DataFrame(data)

df_predicted_label = pd.json_normalize(predicted_df['Entities'])
df_predicted_label['CLEANED_AMOUNT'] = df_predicted_label['AMOUNT'].apply(convert_amounts)
df_predicted_label['CLEANED_DATE'] = df_predicted_label['DATE'].apply(clean_and_parse_date)
predicted_df = predicted_df[['Statement']]

df = pd.concat([predicted_df, df_predicted_label], axis=1)
df.to_excel("./validation_Spacy_trf_19072023_1147.xlsx",index = False)
