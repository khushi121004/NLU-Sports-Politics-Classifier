import re

def clean_text(text):
    text = text.lower() #lowercase
    text = re.sub(r'\d+', '', text) #remove numbers
    text = re.sub(r'[^\w\s]', '', text) #remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() #remove extra spaces
    return text
