# Project Description
This portion of the project simply,
- imports messages and their respective categories in CSV form,
- clean and custom format dataand finally,
- export and save outputs in a SQL lite database for subsequent use.

This scripts is initialized by the following libraries
* import pysftp # library provides sftp connectivity tools
* import pandas as pd # library provides mathematical suit 
* from datetime import datetime as dt #library allows for manipulation of dates and time
* import os # library allows for detection and manipulation of file paths / directories
* from sqlalchemy import create_engine # library allows for creation of sql engine

The Extract, Transform, and Load (ETL) script is to be executed thusly:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

