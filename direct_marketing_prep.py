#!/usr/bin/env python
# coding: utf-8

# In[24]:


def visitpredict(df):
    import pycaret.classification
    nb = pycaret.classification.load_model('C:/Users/owner/Desktop/nb_direct')
    pred = pycaret.classification.predict_model(nb, df)
    return pred

def get_output_schema():
     return pd.DataFrame({
     'ID' : prep_int(),
     'recency' : prep_int(),
     'history_segment' : prep_string(),
     'history' : prep_decimal(),
     'mens' : prep_int(),
     'womens' : prep_int(),
     'zip_code' : prep_string(),
     'newbie' : prep_int(),
     'channel' : prep_string(),
     'segment' : prep_string(),
     'DM_category' : prep_int(),
     'Label' : prep_decimal(),
     'Score' : prep_decimal()
     });

