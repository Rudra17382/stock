import pyarrow
import pandas as pd 
import yfinance as yf


# This is just an example
import sqlite3
import yfinance as yf

# Create database and table
conn = sqlite3.connect('db.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS prices(
        datetime TEXT
        ,open REAL
        ,high REAL
        ,low REAL
        ,close REAL
        ,"adj close" REAL
        ,volume INTEGER
        ,sym TEXT)
''')

# Extract, transform
df = yf.download(tickers='BYND', period = '1d', interval='1m').reset_index()
df['sym'] = sym

# Load
df.to_sql('prices', conn, if_exists='append', index=0)