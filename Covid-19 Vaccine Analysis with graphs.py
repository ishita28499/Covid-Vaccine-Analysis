#!/usr/bin/env python
# coding: utf-8

# In[206]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("C:/Users/ISHITA/Desktop/Assignment/country_vaccinations.csv/country_vaccinations.csv")
data1 = pd.read_csv("C:/Users/ISHITA/Desktop/Assignment/country_vaccinations_by_manufacturer.csv/country_vaccinations_by_manufacturer.csv")


# In[207]:


data.head()


# In[208]:


data1.head()


# In[209]:


import os
import glob


# In[210]:


#for column in data1.columns:
 #   data[column] = data1[column].iloc[0]
#data


# In[211]:


pd.to_datetime(data.date)
data.country.value_counts()


# In[212]:


data.vaccines.value_counts()


# In[213]:


df = data[["vaccines", "country"]]
df.head()


# In[214]:


dict_ = {}
for i in df.vaccines.unique():
  dict_[i] = [df["country"][j] for j in df[df["vaccines"]==i].index]

vaccines = {}
for key, value in dict_.items():
  vaccines[key] = set(value)
for i, j in vaccines.items():
  print(f"{i}:>>{j}")


# In[215]:


#Vaccines used by specefic Country.
import plotly
import plotly.express as px
import plotly.offline as py

vaccine_map = px.choropleth(data, locations = 'iso_code', color = 'vaccines')
vaccine_map.update_layout(height=400, margin=dict(r=0,t=0,l=0,b=0,autoexpand=True))
vaccine_map.show()


# fig = px.choropleth(df.reset_index(), locations="iso_code",
#                     color="total_vaccinations_per_hundred",
#                     color_continuous_scale=px.colors.sequential.Electric,
#                    title= "Total vaccinations per 100")

# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})  #No margin on left, right, top and bottom
# fig.show()


# In[216]:


len(np.unique(data['country']))


# In[217]:


clean_data1 = data.dropna(subset=['people_fully_vaccinated'])


# In[218]:


len(np.unique(clean_data1['country']))


# In[219]:


list_countries = np.unique(clean_data1['country'])
latest_vaccinations = np.array([])


# In[220]:


for country in list_countries:
    latest_vaccinations = np.append(latest_vaccinations,
                                    clean_data1[clean_data1['country']==country] 
                                    .iloc[-1]['people_fully_vaccinated'])


# In[221]:


fully_vaccinated = pd.DataFrame({'country': list_countries,
                                'people_fully_vaccinated': latest_vaccinations})


# In[222]:


ranking1 = fully_vaccinated.sort_values(by=['people_fully_vaccinated'],ascending=False,ignore_index=True)


# In[223]:


ranking1


# In[224]:


np.unique(data1['vaccine']) #list with vaccines used


# In[225]:


data.columns


# In[226]:


data1.columns


# In[227]:


clean_data1_h = data.dropna(subset=['people_fully_vaccinated_per_hundred'])
latest_ratio = np.array([]) # Here, we will save the latest values for 'people_fully_vaccinated_per_hundred'

for country in list_countries:
    latest_ratio = np.append(latest_ratio,clean_data1_h[clean_data1_h['country']==country].iloc[-1]['people_fully_vaccinated_per_hundred'])

ratio_fully_vaccinated = pd.DataFrame({'country': list_countries,'people_fully_vaccinated_per_hundred': latest_ratio})

ranking2 = ratio_fully_vaccinated.sort_values(by=['people_fully_vaccinated_per_hundred'],ascending=False,ignore_index=True)


# In[228]:


ranking2


# In[229]:


np.unique(data1['location'])


# In[230]:


types_EU = np.unique(data1[data1['location']=='European Union']['vaccine'])


# In[231]:


types_EU


# In[232]:


cova = data1[(data1['location']=='European Union') & 
             (data1['dates']=='07-03-2022') & 
             (data1['vaccine']=='Covaxin')]['total_vaccinationss'].iloc[0]

jandj = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Johnson&Johnson')]['total_vaccinationss'].iloc[0]

mod = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Moderna')]['total_vaccinationss'].iloc[0]

nova = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Novavax')]['total_vaccinationss'].iloc[0]

oxfast = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Oxford/AstraZeneca')]['total_vaccinationss'].iloc[0]

pfibio = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Pfizer/BioNTech')]['total_vaccinationss'].iloc[0]

sinobei = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Sinopharm/Beijing')]['total_vaccinationss'].iloc[0]

sinov = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Sinovac')]['total_vaccinationss'].iloc[0]

sput = data1[(data1['location']=='European Union') & 
              (data1['dates']=='07-03-2022') & 
              (data1['vaccine']=='Sputnik V')]['total_vaccinationss'].iloc[0]

numbers_EU = [cova,jandj,mod,nova,oxfast,pfibio,sinobei,sinov,sput]


# In[240]:


for i in range(len(types_EU)):
    print(types_EU[i]+': '+str(numbers_EU[i]))


# In[234]:


plt.figure(figsize=(10,6))
plt.bar(types_EU,numbers_EU)
plt.xticks(types_EU,rotation='vertical')
plt.yticks(numbers_EU)
plt.xlabel('Type of vaccine')
plt.ylabel('Number of vaccines')
plt.title('COVID-19 vaccines administered in the European Union until March 7, 2022')
plt.show()


# In[235]:


#The date a country reaches 80% full vax, quickest countries to reach
df_1st = data.loc[data['people_fully_vaccinated_per_hundred']>80,['country','date','people_fully_vaccinated_per_hundred']]
df_1st.groupby(['country']).min().sort_values(by = 'date')


# In[236]:


#The highest vax level in a country currently, only countries w/ vax >80%
#Max selects the most recent date of a country, we only select countries with vax >80%
q1 = data.groupby(['country'])[['date', 'people_fully_vaccinated_per_hundred']].max()
q1.loc[q1['people_fully_vaccinated_per_hundred']>80,].sort_values(by = 'people_fully_vaccinated_per_hundred', ascending = False)


# In[237]:


#Countries with fully vax <20%
q2 = data.groupby(['country'])[['date', 'people_fully_vaccinated_per_hundred']].max()
# under20 = 
q2.loc[q2['people_fully_vaccinated_per_hundred']<20,].sort_values(by = 'people_fully_vaccinated_per_hundred', ascending = False)


# In[199]:


#Show vaccination rates of G7 countries
df_plot = data.loc[:,['country', 'date', 'people_fully_vaccinated_per_hundred']]
grpA = ['France', 'Canada', 'United States', 'Germany', 'Italy', 'Japan', 'United Kingdom']
df_A = df_plot[df_plot['country'].isin(grpA)]

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
fig, axes = plt.subplots(figsize = (15,7))
sns.lineplot(x = df_A['date'], y = 'people_fully_vaccinated_per_hundred', hue = 'country', data = df_A, ax = axes, linewidth = 2)
# axes.xaxis.set_major_formatter(DateFormatter("%b-%y"))
axes.xaxis.set_major_locator(mdates.MonthLocator(interval = 2))
axes.set_xlabel("")
axes.set_ylabel("% Fully Vaccinated")
axes.set_title("G7 countries comparison - % Fully Vaccinated")
axes.legend(title ="", prop = {'size':16})


# In[201]:


#Vaccination rates of SEA countries
df_plot = data.loc[:,['country', 'date', 'people_fully_vaccinated_per_hundred']]
grpA = ['Singapore', 'Philippines', 'Malaysia', 'Thailand', 'Brunei', 'India', 'Indonesia']
df_A = df_plot[df_plot['country'].isin(grpA)]

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
fig, axes = plt.subplots(figsize = (12,7))
sns.lineplot(x = data["date"], y = 'people_fully_vaccinated_per_hundred', hue = 'country', data = df_A, ax = axes, linewidth = 2)
axes.xaxis.set_major_locator(mdates.MonthLocator(interval = 2))
axes.set_xlabel("")
axes.set_ylabel("% Fully Vaccinated")
axes.set_title("SEA countries comparison % Fully Vaccinated")
axes.legend(title ="", prop = {'size':15.1})


# In[60]:


#List of countries we will compare
countries = ['Hong Kong', 'Japan', 'Spain', 'European Union', 'United States', 'Chile', 'South Korea']


# In[61]:


#Vaccines used in each country
vax_grp = data1.groupby(['location', 'vaccine'])[['dates', 'total_vaccinationss']]
max_vax = vax_grp.max().reset_index()
max_vax.loc[max_vax['location'].isin(countries),:].sort_values(by = ['location','total_vaccinationss'], ascending = [True, False])


# In[62]:


#Most popular vaccines in the 42 countries
data1.groupby(['vaccine'])[['dates', 'total_vaccinationss']].max().sort_values(by = 'total_vaccinationss', ascending = False)


# In[63]:


df_India = data[data['iso_code'] == 'IND'].copy()
df_India


# In[64]:


df_India.drop(df_India.index[df_India['total_vaccinations']==0],inplace=True)


# In[65]:


#Plot total vaccinations as a function of date
data['date'] = pd.to_datetime(data['date'])
# h=data['date'].head()
# i=data['date'].tail()
plt.figure(figsize=(18,6))
sns.lineplot( x=df_India["date"], y=df_India["daily_vaccinations"])
plt.title("Total vaccinations in India")
plt.xticks(rotation=45)
plt.show()


# In[66]:


#Plot daily vaccinations as a function of date
plt.figure(figsize=(16,8))
sns.lineplot(x=df_India["date"], y=df_India["total_vaccinations"])
plt.xticks(rotation=90)
plt.title("Daily vaccinations in India")


# In[67]:


#Sort by total vaccinations delivered by countries and group by vaccines. 
vacc_names_by_country = data.groupby('vaccines').max().sort_values('total_vaccinations', ascending=False)
vacc_names_by_country.head()


# In[68]:


#Get the top 10 vaccines by country for easy plotting
vacc_names_by_country = vacc_names_by_country.iloc[:10]
vacc_names_by_country


# In[69]:


#Reset index to move vaccines from being index to a column. 
#This makes it easy for us to plot using Seaborn, especially if we want to sort by country. 
vacc_names_by_country=vacc_names_by_country.reset_index()
vacc_names_by_country


# In[70]:


plt.figure(figsize=(12,8))

sns.barplot(data = vacc_names_by_country, x='vaccines', y = 'total_vaccinations', hue = 'country', dodge=False)
plt.xticks(rotation=90)


# In[71]:


fig = px.choropleth(data.reset_index(), locations="iso_code",
                    color="total_vaccinations_per_hundred",
                    color_continuous_scale=px.colors.sequential.Electric,
                   title= "Total vaccinations per 100")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})  #No margin on left, right, top and bottom
fig.show()


# In[147]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from pywaffle import Waffle
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
init_notebook_mode(connected=True)
import warnings
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# In[148]:


#Plotting the heatmap of correlation between features

corr = data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


# In[149]:


#What vaccines are used in each country?

country_vaccine = data.groupby(["country", "iso_code", "vaccines"])['total_vaccinations', 
                                                                       'total_vaccinations_per_hundred',
                                                                        'daily_vaccinations',
                                                                        'daily_vaccinations_per_million',
                                                                        'people_vaccinated',
                                                                        'people_vaccinated_per_hundred',
                                                                        'people_fully_vaccinated', 'people_fully_vaccinated_per_hundred'
                                                                        ].max().reset_index()
                                                                        
country_vaccine.columns = ["Country", "iso_code", "Vaccines", "Total vaccinations", "Percent", "Daily vaccinations", 
                           "Daily vaccinations per million", "People vaccinated", "People vaccinated per hundred",
                           'People fully vaccinated', 'People fully vaccinated percent']


# In[150]:


vaccines = country_vaccine.Vaccines.unique()
for v in vaccines:
    countries = country_vaccine.loc[country_vaccine.Vaccines==v, 'Country'].values
    print(f"Vaccines: {v}: \nCountries: {list(countries)}\n") 


# In[151]:


#drop total vaccinations missing data, as without this value any raw doesn't make much sense.

data = data.drop(data[data.total_vaccinations.isna()].index)
data= data.drop(data[data.people_vaccinated.isna()].index)
data.isna().sum()


# In[152]:


corr = data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


# In[153]:


#So, we will fill the missing values with the difference of these column's mean values.

difference = data.total_vaccinations.mean() - data.people_vaccinated.mean()
difference_per_hundred = data.total_vaccinations_per_hundred.mean() - data.people_vaccinated_per_hundred.mean()

data.people_vaccinated = data.people_vaccinated.fillna(data.total_vaccinations - difference)
data.people_vaccinated_per_hundred = data.people_vaccinated_per_hundred.fillna(data.total_vaccinations_per_hundred - difference_per_hundred)

data.isna().sum()


# In[154]:


#for other feature just fill missing values with zeros

data.daily_vaccinations = data.daily_vaccinations.fillna(0)
data.daily_vaccinations_per_million = data.daily_vaccinations_per_million.fillna(0)
data.people_fully_vaccinated = data.people_fully_vaccinated.fillna(0)
data.people_fully_vaccinated_per_hundred = data.people_fully_vaccinated_per_hundred.fillna(0)
data.daily_vaccinations_raw = data.daily_vaccinations_raw.fillna(0) 

data.isna().sum()


# In[155]:


NavidData = data.copy()

NavidData.rename(columns={'country':'Country',
                          'iso_code':'IsoCode',
                          'date':'Date',
                          'total_vaccinations': 'TotalVaccinations', 
                          'people_vaccinated': 'PeopleVaccinated',
                          'people_fully_vaccinated': 'PeopleFullyVaccinated',
                          'daily_vaccinations_raw': 'DailyVaccinationsRaw',
                          'daily_vaccinations': 'DailyVaccinations',
                          'total_vaccinations_per_hundred': 'TotalVaccinationsPerHundred',
                          'people_vaccinated_per_hundred':'PeopleVaccinatedPerHundred',
                          'people_fully_vaccinated_per_hundred': 'PeopleFullyVaccinatedPerHundred',
                          'daily_vaccinations_per_million': 'DailyVaccinationsPerMillion',
                          'vaccines':'Vaccines',
                          'source_name':'SourceName',
                          'source_website':'SourceWebsite'}, inplace=True) 


# In[161]:


#Which country is vaccinating its citizens the fastest?

cols = ['Country', 'TotalVaccinations', 'IsoCode', 'Vaccines','TotalVaccinationsPerHundred']

vacc_amount = NavidData[cols].groupby('Country').max().sort_values('TotalVaccinations', ascending=False).dropna(subset=['TotalVaccinations'])
vacc_amount = vacc_amount.iloc[:10]

vacc_amount = vacc_amount.sort_values('TotalVaccinationsPerHundred', ascending=False)

plt.figure(figsize=(10, 7))
plt.bar(vacc_amount.index, vacc_amount.TotalVaccinationsPerHundred ,color=['black', 'red', 'green', 'blue', 'orange','pink', 'yellow', 'purple', 'grey', 'brown'])

plt.ylabel('Number of vaccinated people per hundred')
plt.xlabel('Countries')
plt.show()


# In[204]:


#What are the different categories of vaccines offered?

plt.figure(figsize=(15,15))
grp = ['Country', 'TotalVaccinations', 'IsoCode', 'Vaccines']
vacc_no = NavidData[grp].groupby('Vaccines').max().sort_values('TotalVaccinations', ascending=False).dropna(subset=['TotalVaccinations'])

plt.bar(vacc_no.index, vacc_no.TotalVaccinations , color ='m')
plt.title('Various categories of COVID-19 vaccines offered')
plt.xticks(rotation = 90)
plt.ylabel('Number of vaccinated citizens (per 10 Million)')
plt.xlabel('Vaccines')
plt.show()


# In[203]:


#Percentage of use of different types of vaccines.

data = dict(NavidData['Vaccines'].value_counts(normalize = True).nlargest(10)*100) 
                                 
vaccine = ['Oxford/AstraZeneca', 'Moderna, Oxford/AstraZeneca, Pfizer/BioNTech',
       'Oxford/AstraZeneca, Pfizer/BioNTech',
       'Johnson&Johnson, Moderna, Oxford/AstraZeneca, Pfizer/BioNTech',
       'Pfizer/BioNTech', 'Sputnik V', 'Oxford/AstraZeneca, Sinopharm/Beijing',
       'Sinopharm/Beijing', 'Moderna, Pfizer/BioNTech',
       'Oxford/AstraZeneca, Pfizer/BioNTech, Sinovac'] 

fig = plt.figure( 
    rows=7,
    columns=12,
    FigureClass = Waffle, 
    values = data, 
    title={'label': 'Proportion of Vaccines', 'loc': 'center',
          'fontsize':15},
    colors=("#FF7F0E", "#00B5F7", "#AB63FA","#00CC96","#E9967A","#F08080","#40E0D0","#DFFF00","#DE3163","#6AFF00"),
    labels=[f"{k} ({v:.2f}%)" for k, v in data.items()],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': 2, 'framealpha': 0},
    figsize=(12, 9)
)
fig.show()


# In[84]:


#Total Vaccinations per country grouped by Vaccines.

fig = px.treemap(NavidData,names = 'Country',values = 'TotalVaccinations',
                 path = ['Vaccines','Country'],
                 title="Total Vaccinations per country grouped by Vaccines",
                 color_discrete_sequence =px.colors.qualitative.Set1)
fig.show()


# In[85]:


CountryVaccine = NavidData.groupby(["Country", "IsoCode", "Vaccines"])['TotalVaccinations', 
                                                                       'TotalVaccinationsPerHundred',
                                                                      'DailyVaccinations',
                                                                      'DailyVaccinationsPerMillion',
                                                                      'PeopleVaccinated',
                                                                      'PeopleVaccinatedPerHundred',
                                                                      'PeopleFullyVaccinated', 
                                                                      'PeopleFullyVaccinatedPerHundred'
                                                                      ].max().reset_index()

CountryVaccine.columns = ["Country", "iso_code", "Vaccines", "Total vaccinations", "Percent", "Daily vaccinations", 
                           "Daily vaccinations per million", "People vaccinated", "People vaccinated per hundred",
                           'People fully vaccinated', 'People fully vaccinated percent']


# In[86]:


fig = px.treemap(CountryVaccine, path = ['Vaccines', 'Country'], values = 'Total vaccinations',
                title="Total vaccinations per country, grouped by vaccine scheme")
fig.show()


# In[87]:


#People vaccinated per country, grouped by vaccine scheme

fig = px.treemap(CountryVaccine, path = ['Vaccines', 'Country'], values = 'People vaccinated',
                title="People vaccinated per country, grouped by vaccine scheme")
fig.show()


# In[88]:


# vaccinsation total per country

def draw_trace_bar(data, feature, title, xlab, ylab,color='Blue'):
    data = data.sort_values(feature, ascending=False)
    trace = go.Bar(
            x = data['Country'],
            y = data[feature],
            marker=dict(color=color),
            text=data['Country']
        )
    data = [trace]

    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=45, 
                           zeroline=True, zerolinewidth=1, zerolinecolor='grey',
                           showline=True, linewidth=2, linecolor='black', mirror=True,
                          tickfont=dict(
                            size=10,
                            color='black'),), 
              yaxis = dict(title = ylab, gridcolor='lightgrey', zeroline=True, zerolinewidth=1, zerolinecolor='grey',
                          showline=True, linewidth=2, linecolor='black', mirror=True),
              plot_bgcolor = 'rgba(0, 0, 0, 0)', paper_bgcolor = 'rgba(0, 0, 0, 0)',
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')


# In[89]:


draw_trace_bar(CountryVaccine, 'Total vaccinations', 'Vaccination total per country', 'Country', 'Vaccination total', "Darkgreen" )


# In[90]:


#People vaccinated per hundred Country

draw_trace_bar(CountryVaccine, 'People vaccinated per hundred', 'People vaccinated per hundred per country', 'Country','People vaccinated per hundred', "orange" )


# In[91]:


#vaccination (total vs percentage)

def plot_custom_scatter(df, x, y, size, color, hover_name, title):
    fig = px.scatter(df, x=x, y=y, size=size, color=color,
               hover_name=hover_name, size_max=80, title = title)
    fig.update_layout({'legend_orientation':'h'})
    fig.update_layout(legend=dict(yanchor="top", y=-0.2))
    fig.update_layout({'legend_title':'Vaccine scheme'})
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.show()


# In[92]:


plot_custom_scatter(CountryVaccine, x="Total vaccinations", y="Percent", size="Total vaccinations", color="Vaccines",
           hover_name="Country", title = "Vaccinations (Percent vs. total), grouped per country and vaccines")


# In[93]:


#Total Vaccinations per Country.

trace = go.Choropleth(
            locations = country_vaccine['Country'],
            locationmode='country names',
            z = country_vaccine['Total vaccinations'],
            text = country_vaccine['Country'],
            autocolorscale =False,
            reversescale = True,
            colorscale = 'viridis',
            marker = dict(
                line = dict(
                    color = 'rgb(0,0,0)',
                    width = 0.5)
            ),
            colorbar = dict(
                title = 'Total vaccinations',
                tickprefix = '')
        )

data = [trace]
layout = go.Layout(
    title = 'Total vaccinations per country',
    geo = dict(
        showframe = True,
        showlakes = False,
        showcoastlines = True,
        projection = dict(
            type = 'natural earth'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)


# In[94]:


#Daily Vaccinations per Million per Country.

trace = go.Choropleth(
            locations = country_vaccine['Country'],
            locationmode='country names',
            z = country_vaccine['Daily vaccinations per million'],
            text = country_vaccine['Country'],
            autocolorscale =False,
            reversescale = True,
            colorscale = 'viridis',
            marker = dict(
                line = dict(
                    color = 'rgb(0,0,0)',
                    width = 0.5)
            ),
            colorbar = dict(
                title = 'Daily vaccinations per million',
                tickprefix = '')
        )

data = [trace]
layout = go.Layout(
    title = 'Daily vaccinations per million per country',
    geo = dict(
        showframe = True,
        showlakes = False,
        showcoastlines = True,
        projection = dict(
            type = 'natural earth'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)


# In[95]:


#Prepare Data for Machine Learning.


#Remove unnecessary features.

del NavidData['IsoCode']
del NavidData['Vaccines']
del NavidData['SourceName']
del NavidData['SourceWebsite']


# In[96]:


#Predicting Total Vaccinations for India.

India = NavidData[NavidData.Country.str.contains("India")]


# In[97]:


print ('india data shape:',India.shape)


# In[98]:


India


# In[99]:


India.describe(include='all').T


# In[100]:


#feature scaling

scaler = StandardScaler()
India[['TotalVaccinations','PeopleVaccinated','PeopleFullyVaccinated',
       'DailyVaccinationsRaw','DailyVaccinations','DailyVaccinations',
       'TotalVaccinationsPerHundred','PeopleVaccinatedPerHundred',
       'PeopleFullyVaccinatedPerHundred',
       'DailyVaccinationsPerMillion']] = scaler.fit_transform(India[['TotalVaccinations',
                                                                     'PeopleVaccinated','PeopleFullyVaccinated',
                                                                     'DailyVaccinationsRaw','DailyVaccinations',
                                                                     'DailyVaccinations','TotalVaccinationsPerHundred',
                                                                     'PeopleVaccinatedPerHundred',
                                                                     'PeopleFullyVaccinatedPerHundred',
                                                                     'DailyVaccinationsPerMillion']])


# In[101]:


India


# In[72]:


#Splitting India dataset

IndiaX = India[['PeopleVaccinated','PeopleFullyVaccinated','DailyVaccinationsRaw','DailyVaccinations','DailyVaccinations','TotalVaccinationsPerHundred','PeopleVaccinatedPerHundred','PeopleFullyVaccinatedPerHundred','DailyVaccinationsPerMillion']]   
IndiaY = India[['TotalVaccinations']] #Target


# In[73]:


X_train_I, X_test_I, Y_train_I, Y_test_I = train_test_split(IndiaX,IndiaY, test_size = 0.20, random_state =2,shuffle=False)


# In[92]:


#Extra Tree Model.

EXTRAindia = ExtraTreesRegressor().fit(X_train_I, Y_train_I)


# In[93]:


# EXTRAindia.fit(X_train_I, Y_train_I)


# In[94]:


EXTRAindia.score(X_test_I, Y_test_I)


# In[95]:


#Make Predictions.

Y_pred_I = EXTRAindia.predict(X_test_I)


# In[96]:


print('MAE:',metrics.mean_absolute_error(Y_test_I, Y_pred_I))
print('MSE:',metrics.mean_squared_error(Y_test_I, Y_pred_I))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test_I, Y_pred_I)))


# In[97]:


Y_pred_I


# In[98]:


Y_test_I


# In[ ]:





# In[ ]:




