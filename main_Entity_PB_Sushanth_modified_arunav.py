import os
import re
import ast
import sys
sys.path.append("./depends")
import spacy
import pandas as pd
from dateutil import * 
from datetime import *
from dateutil.parser import *
import json
from tqdm import tqdm
import find_entity as fe
import preprocess as pp
#import get_daterange as gd
#import get_daterange_new_zabi_26062023 as gd
import get_daterange_new_zabi_2_26062023 as gd
from datetime import datetime

class Main:
    def __init__(self):
        #self.model_path = './model_Zayed_20062023/entity_extraction_model/'
        self.model_path = './model/entity_extraction_model/'
        self.unnecessary_chars = ['?','#','XX','!','@','*','{','}',')','(','[',']','external email warning', 'do not click on any attachment or links/url in this email unless sender is reliable','24x7']
        self.amt_dic = {5: ['lacs','lac','lakhs','lakh','lk','lkhs'], 3:['k','thousand'], 7:['cr','crore','crores']}
        # self.date_formats = {'year':['%y','%Y'],'month':['%b','%B', '%B month'],'m-y': ['%Y %B','%B %Y','%Y %b','%b %Y','%B %y']}
        self.date_formats = {'year':['%y','%Y'],'m-y': ['%b','%B', '%B month','%Y %B','%B %Y','%Y %b','%b %Y','%B %y']}
        self.month_str = {'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
        ### LOAD MODEL
        self.nlp = spacy.load(self.model_path)
        ruler = self.nlp.add_pipe("entity_ruler")
        pattern = [{'label':"MobileNo", 'IS_DIGIT':True, 'LENGTH':{"=": 10}, "pattern": [{"TEXT": {"REGEX": r"\b\d{10}\b"}}]},
                    {'label':'SRNumber', "pattern":[{"TEXT":{"REGEX":r'\bSR[0-9]{8,11}|SR\s[0-9]{8,11}\b'}}]}]
        ruler.add_patterns(pattern)

        ##-- MODIFICATION ZABI
    def find_last_years(self, str_date, text):
        match_output = []
        pattern = r"\blast\s+(?:\w+\s?)*?(?:year|years|yrs)\b"
        matches = re.findall(pattern, str(str_date), re.IGNORECASE)
        for index,match in enumerate(matches):
            match_output.append([match])
            split_text = text.split(match)
            pos_indexes = [len(split_text[0]),len(split_text[0])+len(match)]
            match_output[index].extend(pos_indexes)
            
        return match_output
        ##-- END
    def replace_month(self, str_date):
        old_month = ''
        new_month = ''
        for s in str_date.split():
            if s.lower() in month_name_list:
                new_month = random.choice(month_name_list)
                old_month = s
                break
            elif s.lower() in mon_name_list:
                new_month = random.choice(mon_name_list)
                old_month = s
                break
        if old_month == '':
            pass
        str_date = str_date.replace(old_month,new_month)
        return str_date

    def random_date(self, str_date):
        old_str = str_date
        str_date = replace_month(str_date)
        now = parse(str_date, dayfirst=True)
        if old_str == str_date:
            new_month = random.choice(month_list)
            str_date = str_date.replace(str(now.month),new_month)

        new_day = random.choice(day_list)
        new_year = random.choice(year_list)
        str_date = str_date.replace(str(now.year), new_year).replace(str(now.day), new_day)

        # print(old_str, "->>>",str_date)
        return str_date
    
    def filter_date(self,text):
        self.start_text = ['date:' , 'date']
        self.end_text = ['wrote', 'wrote:']
        if 'forwarded message' in text.lower() or 'wrote' in text.lower() or 'date' in text.lower():
            text_list = text.split('\n')
            new_list = text_list
            try:
                for i,t in enumerate(text_list):
                    words_list = t.split()
                    if(words_list != [] and words_list[0].lower() in self.start_text):
                        new_list.pop(i) 
                    elif(words_list != [] and words_list[-1].lower() in self.end_text):
                        if len(words_list) > 2:
                            new_list.pop(i) 
                        else:
                            new_list.pop(i-1)
            except Exception as e:
                pass
            text = '\n'.join(new_list)
        return text

    def remove_card_number(self,text):
        pattern = r"\b\d{4} \d{4} \d{4} \d{4}\b"
        text = re.sub(pattern, "", text)
        return text

    def preprocess_data(self, data):
        #data = self.filter_date(data)
        for junk in self.unnecessary_chars:
            data = data.replace(junk,' ')
        data = data.replace(':',': ')
        data = self.remove_card_number(data)
        data = ' '.join(data.split())
        #data = re.sub(r'x+', "", data)   # commented for testing
        data = re.sub(r'\*+', "", data)
        data = re.sub(r'\ +', " ", data)
        data = data.replace('(','')
        data = data.replace(')','')
        data = data.lower()
        data = pp.remove_headers(data)
        #data = pp.remove_non_ascii(data)
        data = pp.remove_emails_urls(data)

        return data

    def get_range_val(self, range_val, change_attr):
        range_val = range_val.replace(' ','')
        current_day = datetime.now()
        if change_attr == 'month': 
            previous_day = current_day - relativedelta.relativedelta(months=int(range_val))
        elif change_attr == 'year':
            if not range_val.isnumeric():
                split_val = re.split('(\d+)',range_val)
                nums = []
                for i in split_val:
                    if i.isnumeric():
                        nums.append(i)
                if len(nums) == 2:
                    for index,n in enumerate(nums):
                        if len(n) == 2:
                            nums[index] = '20'+n
                    #previous_day = current_day.replace(year=int(n[0]))
                    previous_day = current_day.replace(year=int(nums[0]))
                    #current_day = current_day.replace(year=int(n[1]))
                    current_day = current_day.replace(year=int(nums[1]))
                else:
                    return range_val
            else:
                if int(range_val) == current_day.year:
                    previous_day = current_day
                else:
                    previous_day = current_day.replace(year=current_day.year - int(range_val))
        res =  previous_day.strftime('%d-%m-%Y') + ' to ' + current_day.strftime('%d-%m-%Y')
        return res

    def chk_date_range(self,str_date):
        dates = []
        if '-' in str_date:
            dates = str_date.replace(' ','').split('-')
        elif 'to' in str_date:
            dates = str_date.replace(' ','').split('to')
        dates_list = []
        if len(dates) == 2:
            for d in dates:
                for fmt in self.date_formats['year']:
                    try:
                        now = datetime.strptime(d, fmt)
                        dates_list.append(now.year)
                        break
                    except Exception as e:
                        pass
                ### IF BOTH THE DATES COMPILED THEN ADD START & END OF FINANCIAL YEAR
                if len(dates_list) == 2:
                    now = datetime.now()
                    start_month = now.replace(day=1,month=4,year=dates_list[0])
                    end_month = now.replace(day=31, month=3,year=dates_list[1])
                    res = start_month.strftime('%d-%m-%Y') + ' to ' + end_month.strftime('%d-%m-%Y')
                    return res
                
    def parse_date_format(self,str_date, date_range):
        M_y_format = False
        res = str_date

        if len(str_date) < 4:
            return ''
        ### CHECK IF IT IS FINANCIAL YEAR RANGE
        f_year = self.chk_date_range(str_date)   
        if f_year != None:
            return f_year
                
        for k,v in self.date_formats.items():
            ### CHECK FOR FORMATS FOR MONTH & YEAR
            for fmt in v:
                try:
                    if k == 'm-y':
                        first_day = datetime.strptime(str_date, fmt)
                        ### IF YEAR IS NOT MENTIONED... BY DEFAULT IT TAKES 1900. CONVERT TO CURRENT YEAR
                        if first_day.year == 1900:
                            first_day = first_day.replace(year=datetime.now().year)
                        ### GET LAST DATE OF MONTH
                        nxt_mnth = first_day.replace(day=28) + timedelta(days=4)
                        last_day = nxt_mnth - timedelta(days=nxt_mnth.day)
                        if date_range == 'no':
                            res = first_day.strftime('%d-%m-%Y') + ' to ' + last_day.strftime('%d-%m-%Y')
                        else:
                            if date_range == 'yes1':
                                res = first_day.strftime('%d-%m-%Y')
                            else:
                                res = last_day.strftime('%d-%m-%Y')
                        M_y_format = True
                        break
                except Exception as e:
                    pass
        ### IF ABOVE FORMAT IS 
        if not M_y_format:
            try:
                now = parse(str_date,dayfirst=True)
                res = now.strftime('%d-%m-%Y')
            except:
                if 'month' in str_date.lower():
                    #val = str_date.split()[0]
                    val = str_date.replace("month","")
                    if val.isdigit():
                        res = self.get_range_val(val, 'month')
                    elif val.lower() in self.month_str.keys():
                        res = self.get_range_val(self.month_str[val.lower()], 'month')
                elif 'year' in str_date:
                    #val = str_date.split()[0]
                    val = str_date.replace("year","")
                    res = self.get_range_val(val, 'year')
                else:
                    res = ''
        return res

    def parse_amount_format(self, str_amount, word_index, textdata):
        split_val = re.split('(\d+)',str_amount)
        nums = []
        for i in split_val:
            if i.isnumeric():
                nums.append(i)

        if nums == []:
            textdata = textdata[word_index:]
            textlist = [i for i in textdata.split(' ') if i != '']
            if textlist == []:
                return ''
            str_amount = textlist[0]

#        if str_amount.lower() == 'rs.':
#            textdata = textdata[word_index:]
#            textlist = [i for i in textdata.split(' ') if i != '']
#            str_amount = textlist[0]
        
        res = ''
        str_amount = str_amount.replace(' ','').lower()

        test_list = sum(self.amt_dic.values(), [])

        ## CONVERTING UNITS TO DIGITS
        r = any(ele in str_amount.lower() for ele in test_list) 
        if r:
            for num in sorted(self.amt_dic.keys(), reverse=True):
                x = [ele for ele in list(self.amt_dic[num]) if ele in str_amount.lower()]
                if x:
                    str_amount = str_amount.lower().replace(x[0],'').replace('.','').replace(' ','')+'0'*num
        ### REMOVING ALPHA CHAR
        pattern = r'[0-9.,]+[0-9]'
        new_string = re.findall(pattern,str_amount)
        if new_string:
            res = new_string[0].replace(',','')
            
            if res[0] == '.':
                res = res[1:]
            #if len(res) > 6 or not(res.isdigit()): commented by arunav on 09032023 to fix the bug of amount
            #    res = ''
            
        return res

    def parse_mobileNo(self, mobileNo):
        if len(mobileNo) < 10 or len(mobileNo.replace(' ','')) > 12:
            mobileNo = ''
        return mobileNo

    def false_amount(self,text,final_dict):
        text = text.lower()
        #cc_favorite_words = ['credit','cc','card','debit','no.','number','ending','with','four','last','digits','account']
        cc_favorite_words = ['credit','card','debit','ending','four','digits']
        amount_list = final_dict['Amount']
        if amount_list == '':
            return final_dict

        amount_list = amount_list.split(',')

        reject_list = ['5676766','56161','1800','1080']
        #save_list = []
        
        for index,word in enumerate(str(text).split()):
            for amt in amount_list:
                ignore_this_amt = False
                #if len(str(amt)) == 4 and str(amt).lower() in word.lower() and len(str(amt)) != len(word) and word.isnumeric() == True:
                #    reject_list.append(amt)
                if len(str(amt)) == 4 and str(amt).lower() in word.lower()  and len(str(amt)) != len(word) and "xx" in word:
                    reject_list.append(amt)
                elif len(str(amt)) == 4 and str(amt).lower() in word.lower() and str(amt).isdigit():
                    near_2_lower_limit = max(0,index-2)
                    near_2_upper_limit = min(index+2,len(str(text).split()))
                    neighbour_2_seq = str(text).split()[near_2_lower_limit:near_2_upper_limit]
                    for n_words in neighbour_2_seq:
                        for amt_related_word in ['rs.','rupees','inr','amount','transfer','transferred','pay']:
                            if n_words in amt_related_word and len(n_words) > 1:
                                #save_list.append(amt)
                                ignore_this_amt = True

                    lower_limit = max(0,index-3)
                    upper_limit = min(index+2,len(str(text).split()))
                    neighbour_words_seq = str(text).split()[lower_limit:upper_limit]
                    cc_neighbour_count = 0
                    for n_words in neighbour_words_seq:
                        for cc_fav_word in cc_favorite_words:
                            if cc_fav_word in n_words and len(n_words) > 1:
                                cc_neighbour_count+=1
                    if cc_neighbour_count >= 1 and ignore_this_amt == False:
                        reject_list.append(amt)
                elif str(amt).lower() in word.lower() and '.' in str(amt):
                    if str(text).split()[index-1].lower() == 'version:':
                        reject_list.append(amt)
                elif str(amt).lower() in word.lower() and ':' in word.lower():
                    try:
                        re_out = re.findall(r"\d{1,2}:\d{2}",text)
                    except:
                        re_out = []
                    if len(re_out) == 1:
                        reject_list.append(amt)
                elif len(amt) == 6 and amt.lower() in word.lower():
                    word = word.replace('.','')
                    if word.lower() == amt.lower():
                        reject_list.append(amt)

                    
        good_amt = [i for i in amount_list if i not in reject_list]
        with open('pincode_master_india.txt','r') as fp:
            pincode_data = fp.read()
        pincode_data = [i for i in pincode_data.split('\n')]
        good_amt = list(set(good_amt))

        for amt in good_amt:
            if len(amt) == 6:
                for pins in list(set(pincode_data)):
                    if pins in amt and len(pins) == 6:
                        good_amt.remove(amt)
        good_amt = ', '.join(good_amt)
        final_dict['Amount'] = good_amt
        return final_dict

    def get_entity(self, emailBody):

        ### PREPROCESS 
        clean_data = self.preprocess_data(emailBody)

        ### GET ALL LABELS
        ner_labels = self.nlp.get_pipe('ner').labels
        #print(ner_labels)
        self.all_labels = []
        for n in ner_labels:
            if not n.isupper():
                self.all_labels.append(n)
        self.all_labels.append('SRNumber')
   
        ### EXTRACT ENTITY - MODEL
        doc = self.nlp(clean_data)
        
        res = [[ent.text, ent.label_, ent.start_char, ent.end_char] for ent in doc.ents]
        
        undetected_entity = self.find_last_years(doc,clean_data)
        
        ### FORMAT THE RESULT :: {entity_cls : 'data1, data2'}
        resDict = {r:[] for r in self.all_labels}
        #print("======RES_DICT=======",resDict)
        
        dt_cnt = 0
        for r in res:
            
            # UPDATED
            if r[1] == "Date" and r[0] in [i[0] for i in  resDict[r[1]]]:
                continue
            # END
            resDict[r[1]].append([r[0],r[2],r[3]])
            
        resDict['Date'].extend(undetected_entity)
        #print("resDict",resDict)
        #print("resDict[r[1]]", resDict[r[1]],"\n",[r[0],r[2],r[3]])
                
        final_res = {}
        for k,r in resDict.items():
            #print("--K--",k,"--R--",r)
            #print("LENGTH__R",len(r))
            if k.lower() == 'date':
                # UPDATE
                if len(r) == 1:
                    try:
                        text = clean_data[r[0][1]-20:r[0][1]] +  clean_data[r[0][2]:r[0][2]+20]
                        pattern = r"\d*\s*\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\s*\d*"
                        matches = re.findall(pattern, text)
                        if matches:
                            r.append([matches[0].strip(), r[0][1]-20, r[0][1]])
                    except:
                        pass
                # END

                # UPDATE
                if len(r) == 2:
                    try:
                        d1 = datetime.strptime(self.parse_date_format(r[0][0], 'yes1'), "%d-%m-%Y")
                        d2 = datetime.strptime(self.parse_date_format(r[0][0], 'yes2'), "%d-%m-%Y")
                        d3 = datetime.strptime(self.parse_date_format(r[1][0], 'yes1'), "%d-%m-%Y")
                        d4 = datetime.strptime(self.parse_date_format(r[1][0], 'yes2'), "%d-%m-%Y")
                        print("min max")
                        print(min([d1, d2, d3, d4]))
                        print(max([d1, d2, d3, d4]))
                        dmin = min([d1, d2, d3, d4])
                        dmax = max([d1, d2, d3, d4])
                        final_res.update({'DateRange': dmin.strftime("%d-%m-%Y") + " to " + dmax.strftime("%d-%m-%Y")})
                    except:
                        pass
                if 'DateRange' in final_res:
                    final_res["Date"] = ""
                    continue
                if len(r)> 2:               #----------------------------------LINE NO 428------------------------------#
                    print("PRINTED",r)
                # END
                c = r
                result = []
                range_result = []
                index = 0
                for i,l in enumerate(c):
                    if index > len(r)-1:
                        continue
                    ext = self.parse_date_format(r[index][0], 'no')
                    if index < len(r)-1 and r[index+1][1] - r[index][2] <= 5:   
                        mid_word = clean_data[r[index][2]:r[index+1][1]]
                        if 'to' in mid_word or '-' in mid_word or 'and' in mid_word:
                            date1 = self.parse_date_format(r[index][0], 'yes1')
                            date2 = self.parse_date_format(r[index+1][0], 'yes2')
                            ext = date1+' to '+date2
                            index = index+2
                    else:
                        index = index + 1
                    
                    if ext != '' and ext not in range_result and 'to' in ext:
                        range_result.append(ext)
                    elif ext != '' and ext not in result and 'to' not in ext and 'month' not in ext and 'day' not in ext and 'week' not in ext and 'year' not in ext:
                        result.append(ext)
                    else:
                        for vals in ['month','day','year','week']:
                            if vals in ext.lower():
                                nums = re.findall(r"[0-9]",ext)
                                #print("___NUMS___",nums)
                                if nums != []:
                                    try:
                                        ext_output = gd.generate_date_range(ext.lower(),int(nums[0]))
                                    except:
                                        ext_output = False
                                elif nums == []:
                                    try:
                                        ext_output = gd.generate_date_range(ext.lower(),False)
                                    except:
                                        ext_output = False
                                if ext_output != False:
                                    range_result.append(ext_output)

                if "last month" in ' '.join(emailBody.lower().split()):
                    try:
                        ext_output = gd.generate_date_range("last month",False)
                        range_result.append(ext_output)
                    except Exception as e:
                        ext_output = False

                final_res.update({k:','.join(result)})
                final_res.update({'DateRange':','.join(range_result)})
            else:
                result = []
                for i,l in enumerate(r):
                    ext = l[0]
                    if k.lower() == 'amount':
                        ext = self.parse_amount_format(l[0],l[2],clean_data)
                    elif k.lower() == 'mobileno':
                        ext = self.parse_mobileNo(l[0])
                   
                    if ext != '' and ext not in result:
                        result.append(ext)                
                final_res.update({k:','.join(result)})
            
        ### EXTRACT ENTITY - EXTRA PATTERNS
        pattern_amt, wrong_prediction = fe.get_amount_from_pattern(clean_data)
        model_amt = final_res['Amount'].split(',')
        final_amt = [x for x in  list(set(model_amt + pattern_amt)) if x != '' and x not in wrong_prediction]
        final_res.update({'Amount':','.join(final_amt)})
        final_res = self.false_amount(emailBody,final_res)
        #print("final_res--1",final_res)
        return json.dumps(final_res)

    def predict(self, inputStr):
        # ### GET INPUT DATA   
        emailBody = inputStr
      
        ### EXTRACT ENTITY
        result = self.get_entity(emailBody)
    
        return result

