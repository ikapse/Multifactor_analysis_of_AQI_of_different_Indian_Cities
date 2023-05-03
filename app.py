# Importing the relevant packages

import numpy as np
import pandas as pd

from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import skew
import os
from os.path import exists as file_exists

#matplotlib inline


# creating flask object
app = Flask(__name__)


# creating flask page and declaring methods used
@app.route('/', methods=['GET', 'POST'])
def mainbody():
    # Importing the City_day.csv file and loading it onto a created dataframe 'df'

    df=pd.read_csv('city_day.csv',parse_dates=['Date'])
    df.head()

    df.info()

    # Checking descriptive statistical values related to the dataset such as mean, count, min, max, etc.

    df.describe()



    # Imputing the Dataset

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    df["AQI"] = imp.fit_transform(df[['AQI']])


    rcParams['axes.spines.top']=False
    rcParams['axes.spines.right']=False


    rcParams['figure.dpi']=300

    rcParams['figure.autolayout']=True

    rcParams['font.style']='normal'
    rcParams['font.size']=4

    rcParams['lines.linewidth']=0.7


    rcParams['xtick.labelsize']=4
    rcParams['ytick.labelsize']=4


    # Number 1 - Plotting the average AQI per city
    #Grouping the AQI by city and calculating the average AQI per city

    x=pd.DataFrame(df.groupby(['City'])[['AQI']].mean().sort_values(by='AQI').head(10))
    x=x.reset_index('City')

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(3,1.5))
    sns.barplot(data=x,x='AQI',y='City',orient='h',palette='crest')
    plt.xlabel('Mean AQI')
    plt.title('Top 10 Cleanest Cities of India')
    #plt.show()
    plt.savefig('top_cleanest.png')
    # move file to the static subfolder to follow flask protocol
    if file_exists('static/top_cleanest.png'):
        os.replace('top_cleanest.png', 'static/top_cleanest.png')
    else:
        os.rename('top_cleanest.png', 'static/top_cleanest.png')





     # Number 2 - Most Polluted Cities in India - Top 10
    #Grouping the AQI by city and calculating the average AQI per city

    x=pd.DataFrame(df.groupby(['City'])[['AQI']].mean().sort_values(by='AQI',ascending=False).head(10))
    x=x.reset_index('City')

    #plotting the average AQI per city

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(3,1.5))
    sns.barplot(data=x,x='AQI',y='City',orient='h',palette='rocket')
    plt.xlabel('Mean AQI')
    plt.title('Top 10 Most Polluted Cities of India')
    #plt.show()
    plt.savefig('top_polluted.png')
    # move file to the static subfolder to follow flask protocol
    if file_exists('static/top_polluted.png'):
        os.replace('top_polluted.png', 'static/top_polluted.png')
    else:
        os.rename('top_polluted.png', 'static/top_polluted.png')


    # Segregating the date into Month and Year and forming new columns in the dataframe

    df['Month']=df.Date.dt.month.astype(str)
    df['Year']=df.Date.dt.year.astype(str)


    #Collect unique values for the dropdown
    city = df.City.unique()
    years = df.Year.unique()
    print(city)
    print(years)


    # get the variables from frontend using POST procedure
    city_select = request.form.get('city')
    years_select = request.form.get('years')

    if (city_select==None):
        city_select=city[0]
    


    # Number 3 - Change in the particulate matter over the years
    # Visualizing change in amount of particulate matter and gases over the years

    cols=['PM2.5','PM10','NO','NO2','NOx','NH3',
        'CO','SO2','O3','Benzene','Toluene','Xylene']

    x=df.iloc[:,2:]
    fig=plt.figure(figsize=(3.2,6.5))
    plt.title('Change in Particulate Matter over the years')
    for i,col  in enumerate(cols):
        fig.add_subplot(6,2,i+1)
        sns.lineplot(x='Year',y=col,data=x)
    
    #plt.show()
    plt.savefig('change_matter_overyears.png')
    # move file to the static subfolder to follow flask protocol
    if file_exists('static/change_matter_overyears.png'):
        os.replace('change_matter_overyears.png', 'static/change_matter_overyears.png')
    else:
        os.rename('change_matter_overyears.png', 'static/change_matter_overyears.png')  

    df.head(10)

    
    
    # Change in Ahmedabad AQI over the years
    #Grouping the AQI by year and calculating the average AQI per year

    x=pd.DataFrame(df.groupby(['City','Year'])[['AQI']].mean().sort_values(by=['City','Year']))
    x=x.reset_index(['City','Year'])

    x.head(20)

    # Plotting the average AQI of Ahmedabad over the years

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(3,1.5))
    sns.barplot(data=x[x['City'] == 'Ahmedabad'],y='AQI',x='Year',orient = 'v')
    plt.xlabel('AQI of Ahmedabad over the years')
    plt.savefig('avg_ahmedabad.png')
    # move file to the static subfolder to follow flask protocol
    if file_exists('static/avg_ahmedabad.png'):
        os.replace('avg_ahmedabad.png', 'static/avg_ahmedabad.png')
    else:
        os.rename('avg_ahmedabad.png', 'static/avg_ahmedabad.png')  


    # Plotting the average AQI of Selected City over the years

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(3,1.5))
    sns.barplot(data=x[x['City'] == city_select],y='AQI',x='Year',orient = 'v')
    plt.xlabel('AQI of ' + city_select + ' over the years')
    imgpath = 'static/avg_{}.png'.format(city_select)
    plt.savefig(imgpath)
    #plt.savefig('avg_cityselect.png')
    # move file to the static subfolder to follow flask protocol
    #if file_exists('static/avg_cityselect.png'):
    #    os.replace('avg_cityselect.png', 'static/avg_cityselect.png')
    #else:
    #    os.rename('avg_cityselect.png', 'static/avg_cityselect.png')  



    #correlation analysis

    plt.figure(figsize=(3,2))

    sns.heatmap(df.corr(method='pearson'),
                annot=True,fmt='0.1f',
                robust=True,
                cmap='Reds')
    plt.title('Correlation Analysis among the different particulate matters')
    plt.savefig('correlation_analysis.png')
    # move file to the static subfolder to follow flask protocol
    if file_exists('static/correlation_analysis.png'):
        os.replace('correlation_analysis.png', 'static/correlation_analysis.png')
    else:
        os.rename('correlation_analysis.png', 'static/correlation_analysis.png') 

    #Most Clean Days
    x=pd.DataFrame(df['City'][df['AQI']< 100].value_counts())/pd.DataFrame(df['City'].value_counts())*100
    x=x.rename(columns={'City':'Percentage of Days the AQI level was below 100'})
    x.sort_values(by='Percentage of Days the AQI level was below 100', ascending=False, inplace = True)

    plt.figure(figsize=(3,1.8))
    plt.title('Cleanest Cities based on the percentage of Days the AQI of city falls below 100')
    sns.barplot(x='Percentage of Days the AQI level was below 100',y=x.index,data=x,palette='crest')
    plt.savefig('most_clean_days.png')
    # move file to the static subfolder to follow flask protocol
    if file_exists('static/most_clean_days.png'):
        os.replace('most_clean_days.png', 'static/most_clean_days.png')
    else:
        os.rename('most_clean_days.png', 'static/most_clean_days.png')


    
    #Pre and Post Covid Scenario

    df_cs_19 = df[(df['City']==city_select) & (df['Month'].isin(['4','5','6'])) & (df['Year'] == '2019')]
    #Grouping the AQI by month and calculating the average AQI per month
    x_19=pd.DataFrame(df_cs_19.groupby(['Month'])[['AQI']].mean().sort_values(by=['Month']))
    x_19=x_19.reset_index(['Month'])
    x_19 = x_19.replace(['4','5','6'],['Apr','May','Jun'])


    df_cs_20 = df[(df['City']==city_select) & (df['Month'].isin(['4','5','6'])) & (df['Year'] == '2020')]
    #Grouping the AQI by month and calculating the average AQI per month
    x_20=pd.DataFrame(df_cs_20.groupby(['Month'])[['AQI']].mean().sort_values(by=['Month']))
    x_20=x_20.reset_index(['Month'])
    x_20 = x_20.replace(['4','5','6'],['Apr','May','Jun'])



    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(3,1.5))
    fig.suptitle('Comparing Pre-Covid and Post-Covid AQI levels of {}'.format(city_select),size=5)

    axes[0].set_title('AQI of ' + city_select + ' in 2019 (Pre-Covid)',size=3)
    splot1 = sns.barplot(ax=axes[0], data=x_19,y='AQI',x='Month',orient = 'v')
    splot1.set_xlabel('Month',size=3)
    splot1.set_ylabel('Mean AQI',size=3)
    splot1.set_xticklabels(['Apr','May','Jun'],size=3)
    splot1.set_yticklabels(splot1.get_yticks().round(decimals=0), size=3)

    axes[1].set_title('AQI of ' + city_select + ' in 2020 (Post-Covid)',size=3)
    splot2 = sns.barplot(ax=axes[1], data=x_20,y='AQI',x='Month',orient = 'v')
    splot2.set_xlabel('Month',size=3)
    splot2.set_ylabel('Mean AQI',size=3)
    splot2.set_xticklabels(['Apr','May','Jun'],size=3)
    splot2.set_yticklabels(splot2.get_yticks().round(decimals=0), size=3)

    imgpath_covid = 'static/avg_covid_{}.png'.format(city_select)
    plt.savefig(imgpath_covid)


 
    
    
    # render the template folder with html, etc and pass the variables with paths, strings and values
    return render_template('home.html', topcleanest='static/top_cleanest.png', toppolluted='static/top_polluted.png',
                           mostcleandays='static/most_clean_days.png', correlationanalysis='static/correlation_analysis.png', changematter='static/change_matter_overyears.png',
                           avgcityselect=imgpath, avgcityselect_covid=imgpath_covid,
                           city=city, city_select=city_select)


# to initiate the flask container
if __name__ == '__main__':
    app.run()