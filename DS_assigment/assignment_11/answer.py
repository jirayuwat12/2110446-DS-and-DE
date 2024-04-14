import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout='wide')

# load data
@st.cache_data
def load_data():
    df = pd.read_csv('RainDaily_Tabular.csv')
    df['date'] = pd.to_datetime(df['date'])

    province_df = pd.read_excel('ThepExcel-Thailand-Tambon.xlsx',
                                sheet_name='ProvinceDatabase')
    province_df = province_df[['ProvinceThai', 'ภูมิภาคแบบสี่ภูมิภาค']]
    # map province name in df['province'] to province_df['ภูมิภาคแบบสี่ภูมิภาค']
    province_map = dict(province_df.values)
    df['region'] = df['province'].map(province_map)

    return df

df = load_data()
df['date'] = pd.to_datetime(df['date'])


# init constants
PROVINCES = df['province'].unique()
LAT_BOUNDS = [df['latitude'].min(), df['latitude'].max()]
LON_BOUNDS = [df['longitude'].min(), df['longitude'].max()]
GEO_CONFIG = dict(resolution=50,
                  showcoastlines=True,
                  showland=True,
                  showocean=True,
                  showcountries=True,
                  showsubunits=True,
                  subunitcolor='black',
                  countrycolor='black',
                  coastlinecolor='black',
                  landcolor='white',
                  oceancolor='lightblue',
                  projection_type='mercator',
                  lataxis_range=[LAT_BOUNDS[0]-1, LAT_BOUNDS[1]+1],
                  lonaxis_range=[LON_BOUNDS[0]-1, LON_BOUNDS[1]+1],
                  center=dict(lat=0.5*(LAT_BOUNDS[0]+LAT_BOUNDS[1]),
                              lon=0.5*(LON_BOUNDS[0]+LON_BOUNDS[1])))


# title
st.title('Rain amount dashboard by Jirayuwat')
st.write('This dashboard shows the daily rain amount in Thailand.')


# sidebar
st.sidebar.header('Options and filters')

province = st.sidebar.multiselect('Filter province', PROVINCES)
if not province:
    province = PROVINCES

date_range = st.sidebar.date_input('Select date range',
                                   [df['date'].min(), df['date'].max()],
                                   min_value=df['date'].min(),
                                   max_value=df['date'].max())
date_range = pd.to_datetime(date_range)
try:
    date_range = pd.date_range(date_range[0], date_range[1])
except IndexError:
    end_date = date_range[0] + pd.DateOffset(days=1)
    date_range = pd.date_range(date_range[0], end_date)

code = st.sidebar.selectbox('Select meaning of "code" in assignment', ['Station code', 'Program'], index=0)


# filter data
filtered_df = df[(df['province'].isin(province)) & (df['date'].isin(date_range))]


# rain amount by date
st.write('### Rain amount by date')
fig = px.line(filtered_df.groupby('date')['rain'].sum().reset_index(),
                x='date',
                y='rain',
                title='Sum rain amount by date',
                labels={'rain': 'Rain amount (mm)', 'date': 'Date'})
st.plotly_chart(fig, use_container_width=True)


# rain amount by province
col = st.columns(2)
with col[1]:
    st.write('### Map of Thailand rain amount')
    grouped_df = filtered_df.copy()
    grouped_df['latitude'] = (grouped_df['latitude'] // 0.2) * 0.2
    grouped_df['longitude'] = (grouped_df['longitude'] // 0.2) * 0.2
    grouped_df = grouped_df.groupby(['latitude', 'longitude', 'province'])['rain'].sum().reset_index()
    grouped_df['ratio'] = (grouped_df['rain'] / grouped_df['rain'].max())*20
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=grouped_df['longitude'],
        lat=grouped_df['latitude'],
        mode='markers',
        marker=dict(size=grouped_df['ratio'], color=grouped_df['rain'], colorscale='Viridis', colorbar=dict(title='Rain amount (mm)')),
        text=grouped_df['province'],
    ))
    fig.update_layout(geo=GEO_CONFIG,
                        width=400,
                        height=600,
                        title='Map of thai provinces with rain amount',
                        hovermode=False)
    st.plotly_chart(fig)
with col[0]:
    st.write('### Rain amount by province')
    n_provinces = 6
    fig = px.bar(filtered_df.groupby('province')['rain'].sum().sort_values(ascending=False).head(n_provinces).reset_index()[::-1], 
                    x='rain', 
                    y='province', 
                    title=f'{n_provinces} provinces by most rain amount',
                    labels={'rain': 'Rain amount (mm)', 'province': 'Province'})
    fig.update_layout(width=400, height=300)
    st.plotly_chart(fig)

    fig = px.bar(filtered_df.groupby('province')['rain'].sum().sort_values(ascending=True).head(n_provinces).reset_index()[::-1], 
                    x='rain', 
                    y='province', 
                    title=f'{n_provinces} provinces by least rain amount',
                    labels={'rain': 'Rain amount (mm)', 'province': 'Province'})
    fig.update_layout(width=400, height=300)
    st.plotly_chart(fig)


# rain amount by region
st.write('### Rain amount by region')
cols = st.columns(2)
with cols[0]:
    fig = px.pie(filtered_df.groupby('region')['rain'].sum().reset_index(),
                    names='region',
                    values='rain',
                    title='Rain amount by region',
                    color='region',
                    labels={'rain': 'Rain amount (mm)', 'region': 'Region'})
    st.plotly_chart(fig, use_container_width=True)
with cols[1]:
    # plot bar chart of rain amount by date and color by region
    fig = px.bar(filtered_df.groupby(['date', 'region'])['rain'].sum().reset_index(),
                    x='date',
                    y='rain',
                    color='region',
                    title='Rain amount by date and region',
                    labels={'rain': 'Rain amount (mm)', 'date': 'Date', 'region': 'Region'})
    st.plotly_chart(fig, use_container_width=True)


# text summary
rain_date = filtered_df.groupby('date')['rain'].sum()
most_rain_date = rain_date.idxmax().strftime('%d/%m/%Y')
most_rain_amount = rain_date.max()
least_rain_date = rain_date.idxmin().strftime('%d/%m/%Y')
least_rain_amount = rain_date.min()

province_date_rain = filtered_df.groupby(['province', 'date'])['rain'].sum()[lambda x: x==0].reset_index()['province'].value_counts()
most_no_rain_days = province_date_rain.max()
most_no_rain_provinces = ', '.join(province_date_rain[province_date_rain == province_date_rain.max()].index)
least_no_rain_days = province_date_rain.min()
least_no_rain_provinces = ', '.join(province_date_rain[province_date_rain == province_date_rain.min()].index)

st.write(f'### Summary')

if len(province) != 1:
    mesg = f'''
From date {date_range[0].strftime('%d/%m/%Y')} to {date_range[-1].strftime('%d/%m/%Y')} and selected {len(filtered_df['province'].unique()):3,} provinced, the total rain amount in Thailand is {filtered_df['rain'].sum():3,.3f} mm. The date with the most rain amount is {most_rain_date}({most_rain_amount:3,.3f} mm.) and the least rain amount is {least_rain_date}({least_rain_amount:3,.3f} mm.). The province that has the most no-rain day is {most_no_rain_provinces} with {most_no_rain_days} days. Although, the province that has the least no-rain day is {least_no_rain_provinces} with {least_no_rain_days} days.
    '''.strip()
else:
    mesg = f'''
The {province[0]} province has {filtered_df['rain'].sum():3,.3f} mm of rain amount from date {date_range[0].strftime('%d/%m/%Y')} to {date_range[-1].strftime('%d/%m/%Y')}. The date with the most rain amount is {most_rain_date}({most_rain_amount:3,.3f} mm.) and the least rain amount is {least_rain_date}({least_rain_amount:3,.3f} mm.). 
    '''.strip()
st.write(mesg)


# code
st.write('### Code')
if code == 'Station code':
    code = filtered_df[['province', 'name', 'code']]
    code = code.drop_duplicates().sort_values('province').reset_index(drop=True)
    st.write(code)
else:
    code = open('answer.py').read()
    st.code(code, language='python')
