#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np


# ### Basic string processing

# In[2]:


import csv

job_postings = {}

with open('./data/sap_it_test_ops.txt', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row)>2: 
            job_postings[int(row[0])] = [row[1:]]
        else:
            continue


# In[3]:


job_postings = pd.DataFrame.from_dict(job_postings, orient='index', columns=['raw_skill'])
job_postings.head()


# In[4]:


job_postings.info()


# In[5]:


def removePunc(skill):
    pattern = r'.*[(\/)].*'
    temp = re.findall(pattern, skill)
#     print(temp)
# temp[0] != 'r/3', 'sap r/3', 'sap erp r/3', 'erp r/3'...
    if len(temp) > 0 and temp[0] != 's/4 hana' and temp[0] != 's/4hana' and temp[0] != 's4/hana' and temp[0] != 's4/ hana':
#         print(skill)
        skill = skill.replace('/',',')
        skill = skill.split(',')
        return skill
    else:
        return [skill]


# In[6]:


# process version info
# test = list(freq.keys())
def removeNum(string):
    if re.match('.* [0-9]+',string):
        return(re.sub(r' [0-9.]+[a-zA-Z]*',r'',string))
    elif re.match('[0-9]+',string):
        return ''
    else:
        return string

#filter out only contains numbers


# In[7]:


#get_ipython().run_cell_magic('time', '', "\n# deduplication\n# skill_all = []\n
#
skill_processed = []
for row in job_postings.raw_skill.values:
    skill = []
    for i in row:
        i = i.lower().strip().replace(' / ','/').replace(' /','/').replace('/ ','/').replace('_',' ').replace('-',' ')

        i = removeNum(i)
        i = i.replace('(',',').replace(')',',')
        if (',') in i:
            temp = i.split(',')
            for j in temp:
                skill += removePunc(j)
        else:
            skill += removePunc(i)
    skill = list(set(skill))
    skill_processed.append(skill)


job_postings['skill'] = skill_processed
job_postings.head()


# In[9]:


posting_skills = job_postings.explode('skill')
posting_skills.reset_index(inplace=True)
posting_skills = posting_skills.rename(columns = {'index':'posting_id'})
# posting_skills['skill'] = posting_skills['skill'].str.lower()
posting_skills = posting_skills[posting_skills.skill != '…']
posting_skills = posting_skills[posting_skills.skill != '']
posting_skills.head()


# In[10]:


posting_skills.skill.nunique()


# In[11]:


skill_consolidated = pd.read_csv('./data/skills_consolidated_2021_3_22.csv', index_col = 0)
skill_consolidated.head()


# In[12]:


alias = dict(zip(skill_consolidated.alias, skill_consolidated.skill))
alias_posting = posting_skills[posting_skills.skill.isin(alias.keys())]
print(alias_posting)


# In[13]:


#get_ipython().run_cell_magic('time', '', "remap=[]\n\nfor i, s in enumerate(posting_skills['skill']):\n    if s in alias.keys():\n        # Remap the values of the dataframe\n        posting_skills.iloc[i, 2] = alias[s]\n        remap.append(s)\n    if i % 1000 == 0:\n        print(i)\n        \nposting_skills")


# In[14]:


print('number of unique skills in the dataset:', posting_skills.skill.nunique())


# In[15]:


posting_skills.to_csv('processed_skill_posting.csv')
processed_skill_posting = pd.read_csv('processed_skill_posting.csv', index_col = 0)
top_500 = processed_skill_posting['skill'].value_counts()[:500]
top_500.to_csv('top_500_skill_job.csv')


# In[16]:


# 4 hana, s/4 hana, s/4hana, s4/ hana, s4hana, 
# hana, sap hana
# sap hana cloud platform, sap hana cloud, hana cloud
# hana database, hana db
# 4hana finance
# sap netweaver, netweaver
# ariba, aribas, sap ariba
# successfactors, successfactors hcm, successfactors cloud hcm, sap successfactors services cloud
# -sap successfactors saas, sap successfactors, successfactors cloud
# fiori, sap fiori, sap fiori applications, sap fiori apps
# sap vora, hana vora, vora
# sap concur, concur, sap concurs business products, concurs
# cloudfoundry, sap cloud foundry, cloud foundry, cloud foundry[www.cloudfoundry.org
# abap, abap oo, oo abap, sap abap programming language, abap programming, object oriented abap programming, sap abap programming
# hybris, sap hybris, sap hybris marketing solutions suite, sap hybris marketing, sap hybris ecommerce platform, 
# sap hybris cloud, sap marketing cloud
# angular js, angularjs

processed_skill_posting['skill'].replace({'4 hana': 's/4hana', 's/4 hana': 's/4hana', 's4/ hana': 's/4hana', 's4hana': 's/4hana'}, inplace=True)
processed_skill_posting['skill'].replace({'hana': 'sap hana'}, inplace=True)
processed_skill_posting['skill'].replace({'sap hana cloud platform': 'sap hana cloud', 'hana cloud': 'sap hana cloud'}, inplace=True)
processed_skill_posting['skill'].replace({'hana db': 'hana database'}, inplace=True)
processed_skill_posting['skill'].replace({'4hana finance': 's/4hana finance'}, inplace=True)
processed_skill_posting['skill'].replace({'netweaver': 'sap netweaver'}, inplace=True)
processed_skill_posting['skill'].replace({'ariba': 'sap ariba', 'aribas': 'sap ariba'}, inplace=True)
processed_skill_posting['skill'].replace({'successfactors': 'sap successfactors', 'successfactors hcm': 'sap successfactors',
                                 'successfactors cloud hcm': 'sap successfactors', 'successfactors cloud': 'sap successfactors',
                                 'sap successfactors services cloud': 'sap successfactors', 'sap successfactors saas': 'sap successfactors'}, inplace=True)
processed_skill_posting['skill'].replace({'fiori': 'sap fiori', 'sap fiori applications': 'sap fiori', 'sap fiori apps': 'sap fiori'}, inplace=True)
processed_skill_posting['skill'].replace({'sap vora': 'sap hana vora', 'hana vora': 'sap hana vora', 'vora':'sap hana vora'}, inplace=True)
processed_skill_posting['skill'].replace({'concur': 'sap concur', 'sap concurs business products':'sap concur', 'concurs':'sap concur'}, inplace=True)
processed_skill_posting['skill'].replace({'cloudfoundry': 'sap cloudfoundry', 'cloud foundry': 'sap cloudfoundry',
                                 'sap cloud foundry':'sap cloudfoundry', 'cloud foundry[www.cloudfoundry.org': 'sap cloudfoundry'}, inplace=True)
processed_skill_posting['skill'].replace({'abap': 'sap abap', 'abap oo': 'sap abap',
                                 'oo abap': 'sap abap', 'sap abap programming language': 'sap abap',
                                 'abap programming': 'sap abap', 'object oriented abap programming':'sap abap',
                                 'sap abap programming':'sap abap'}, inplace=True)
processed_skill_posting['skill'].replace({'hybris': 'sap hybris', 'sap hybris marketing solutions suite': 'sap hybris',
                                 'sap hybris marketing': 'sap hybris', 'sap hybris ecommerce platform': 'sap hybris'}, inplace=True)
processed_skill_posting['skill'].replace({'angular js': 'angularjs'}, inplace=True)

processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != ' ']
processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != 's']
processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != 'o']


# In[17]:


processed_skill_posting['skill'].replace({"java platform, enterprise edition": 'java ee'}, inplace=True)
processed_skill_posting['skill'].replace({'phyton': 'python'}, inplace=True)
processed_skill_posting['skill'].replace({'4hana cloud': 's/4 hana cloud', '4 hana cloud': 's/4 hana cloud'}, inplace=True)
processed_skill_posting['skill'].replace({'automated tests': 'automated testing', 'automation testing': 'automated testing'}, inplace=True)
processed_skill_posting['skill'].replace({'phyton': 'python'}, inplace=True)
processed_skill_posting['skill'].replace({'4 hana finance': 's/4hana finance'}, inplace=True)
processed_skill_posting['skill'].replace({'4hana fiori': 's/4hana fiori'}, inplace=True)
processed_skill_posting['skill'].replace({'4 ux': 's/4 ux'}, inplace=True)
processed_skill_posting['skill'].replace({'4hana': 's/4hana'}, inplace=True)
processed_skill_posting['skill'].replace({'4 technology': 's/4 technology'}, inplace=True)

processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != '3   s']
processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != '4']
processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != '4g']
processed_skill_posting = processed_skill_posting[processed_skill_posting.skill != '&']


# In[18]:


processed_skill_posting.to_csv('processed_skill_posting.csv')
# processed_skill_posting = pd.read_csv('processed_skill_posting.csv', index_col = 0)
top_500 = processed_skill_posting['skill'].value_counts()[:500]
top_500.to_csv('top_500_skill_job.csv')


# In[19]:


filtered_processed_skill_posting = processed_skill_posting[processed_skill_posting['skill'].map(processed_skill_posting['skill'].value_counts()) >= 10]
filtered_processed_skill_posting['skill'].value_counts()


# In[20]:


filtered_processed_skill_posting.to_csv('filtered_processed_skill_posting.csv')


# ### Construct data files for skill graph model training

# In[21]:


filtered_processed_skill_posting = pd.read_csv('./data/filtered_processed_skill_posting.csv', index_col = 0)
filtered_processed_skill_posting.head()


# In[22]:


skill_posting_relation = filtered_processed_skill_posting.astype({'posting_id': 'category', 'skill': 'category'})
skill_posting_relation.dtypes


# In[23]:


skill_posting_relation.drop('raw_skill', axis = 1, inplace = True)
skill_posting_relation.head()


# In[24]:


skill_posting_relation['posting_ID'] = skill_posting_relation['posting_id'].cat.codes
skill_posting_relation['skill_ID'] = skill_posting_relation['skill'].cat.codes
skill_posting_relation.head()


# In[25]:


skill_posting_relation.info()


# In[26]:


# create entity tables
skill = skill_posting_relation.drop(['posting_ID', 'posting_id'], axis=1)
skill.drop_duplicates(inplace = True)
skill.to_csv('skill.csv', index=False)
skill.head()


# In[27]:


posting = skill_posting_relation.drop(['skill', 'skill_ID'], axis=1)
posting.drop_duplicates(inplace = True)
posting.to_csv('posting.csv', index=False)
posting.head()


# In[28]:


skill_posting_relation.to_csv('skill_posting_relation.csv', index=False)
skill_posting_relation.head()


# ### Preparation for labelling

# In[29]:


tech_skills = pd.read_csv('./skill_data/tech_skills.csv', index_col = 0)
tech_skills.head()


# In[30]:


tech_skills_cat = tech_skills[['Example', 'Commodity Title']]
tech_skills_cat.head()


# In[31]:


tech_skills_cat.drop_duplicates(inplace = True)
len(tech_skills_cat.Example)


# In[32]:


# !pip install fuzzywuzzy
# !pip install python-Levenshtein
from fuzzywuzzy import process, fuzz


# In[36]:


#().run_cell_magic('time', '', "\nskill_alias = {}\nscorers = [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio]\ntarget_skills = tech_skills_cat.Example.values\n\nfor s in skill.skill.values:\n    skill_alias[s] = set()\n    for scorer in scorers:\n        Ratios = process.extract(s, target_skills, scorer = scorer, limit = 20)\n        r = [ts for ts, p in Ratios if p >= 80]\n#         r = [ts for ts, p in Ratios if p > 80 and ts != 'c' and ts != 'c ++' and ts != 'c++']\n        skill_alias[s].update(r) \n#     print(s, skill_alias[s])")
skill_alias = {}
scorers = [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio]
target_skills = tech_skills_cat.Example.values

for s in skill.skill.values:
    skill_alias[s] = set()
    for scorer in scorers:
        Ratios = process.extract(s, target_skills, scorer = scorer, limit = 20)
        r = [ts for ts, p in Ratios if p >= 80]
#         r = [ts for ts, p in Ratios if p > 80 and ts != 'c' and ts != 'c ++' and ts != 'c++']
        skill_alias[s].update(r)

# In[37]:


#skill_alias


# In[17]:


skills_cat_dict = pd.Series(tech_skills_cat['Commodity Title'].values,index=tech_skills_cat.Example).to_dict()
skills_cat_dict


# In[ ]:


with open('skill_alias.json', 'wb') as fp:
    pickle.dump(skill_alias, fp)
with open('skills_cat_dict.json', 'wb') as fp:
    pickle.dump(skills_cat_dict, fp)


# ### Labelling - process restart from here

# In[9]:


import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np


# In[38]:


with open('./data/skill_alias.json', 'rb') as fp:
    skill_alias = pickle.load(fp)
# print(skill_alias)
print(type(skill_alias))

with open('./data/skills_cat_dict.json', 'rb') as fp:
    skills_cat_dict = pickle.load(fp)
# print(skills_cat_dict)
print(type(skills_cat_dict))


# In[97]:


from ipywidgets import interact, interactive, fixed, interact_manual
import webbrowser

term = 'crucible devops'

def f(x):
    if x:
        url = "https://www.google.com.tr/search?q={}".format(term)
        webbrowser.open_new_tab(url)

interact(f, x=False)


# In[93]:


@interact_manual
def f(target_skill=''):
    print(target_skill)


# In[ ]:


# categories1: onet, discard, others, tech general, language, sap product/SAP, soft skills... 
# categories2: data science, product/project management, devops, cloud, security... 


# In[126]:


from IPython.display import clear_output
from ipywidgets import interact
import webbrowser

restart = input('restart? yes or no: ')

if restart == 'yes':
    with open('skill_cat_v1.json', 'rb') as fp:
        skill_cat = pickle.load(fp)
    with open('skill_targetskill_v1.json', 'rb') as fp:
        skill_targetskill = pickle.load(fp)
else:
    skill_cat = {}
    skill_targetskill = {}
    
for i, (k, v) in enumerate(skill_alias.items()):
    if k in skill_cat.keys():
        continue
    else:
        print('skill_num:', i)
        print('skill:', k)
        print('skill_o*net:', v)
        search = input('search? o or x: ')
        if search == 'o':
            url = "https://www.google.com.tr/search?q={}".format(k)
            webbrowser.open_new_tab(url)

        target_skill = input('choose target skill:')
        if target_skill in skills_cat_dict.keys():
            cat = skills_cat_dict[target_skill]
            skill_cat[k] = cat
            skill_targetskill[k] = target_skill
        else:
            skill_cat[k] = target_skill

        if i % 5 == 0:
            with open('skill_cat_v1.json', 'wb') as fp:
                pickle.dump(skill_cat, fp)
            with open('skill_targetskill_v1.json', 'wb') as fp:
                pickle.dump(skill_targetskill, fp)

        clear_output(wait=True)


# ### Skill Normalization - process restart from here

# In[5]:


skill = pd.read_csv('./data/skill.csv')

with open('./data/skill_cat_v1.json', 'rb') as fp:
        skill_cat = pickle.load(fp)


# In[6]:


skill.head()


# In[9]:


skill_onet = pd.DataFrame.from_dict(skill_cat, orient='index', columns=['onet'])
skill_onet.head()


# In[12]:


skill_onet = skill_onet.iloc[:-3,:]
skill_onet.reset_index(inplace = True)
skill_onet.rename(columns = {'index': 'skill'}, inplace = True)
skill_onet.info()


# In[14]:


skill['onet'] = skill_onet.onet.values
skill


# In[15]:


skill.to_csv('skill.csv', index=False)


# In[16]:


skill.groupby(['onet'])['skill'].count()


# In[26]:


from IPython.display import clear_output
cat_discard = ['discard', 'others', 'soft skills', 'language', 'tech general']

restart = input('restart? yes or no: ')

if restart == 'yes':
    with open('skill_normalized.json', 'rb') as fp:
        skill_normalized = pickle.load(fp)
else:
    skill_normalized = {}

scorers = [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio]
for onet, s in skill.groupby(['onet'])['skill']:
    if len(s) > 10 and onet not in cat_discard:
        for a in s:
            if a not in skill_normalized.keys():
                alias = set()
                for scorer in scorers:
                    Ratios = process.extract(a, s, scorer = scorer, limit = 20)
                    r = [ts for ts, p, i in Ratios if p >= 80]
                    alias.update(r)
                alias.difference_update(skill_normalized.keys())
                print(a + ': ', alias)
                n = input('choose target skill:')
                skill_normalized[a] = n
                if len(alias) > 1:
                    for rest in alias:
                        if rest != a:
                            print(rest)
                            n = input('choose target skill:')
                            if n != '':
                                skill_normalized[rest] = n               
            clear_output(wait=True)

        with open('skill_normalized.json', 'wb') as fp:
            pickle.dump(skill_normalized, fp)

