from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='COVID-19 | Data Analytics Portfolio', page_icon='🦠', layout='wide')
ROOT = Path(__file__).resolve().parent
st.markdown('''<style>.stApp{background:#081827;color:#eaf4ff}.block-container{padding-top:1.4rem}h1,h2,h3{color:#edf7ff}[data-testid="stSidebar"]{background:#0c2238}div[data-testid="stMetric"]{background:#102b45;padding:16px;border:1px solid #245378;border-radius:12px}</style>''', unsafe_allow_html=True)

@st.cache_data
def load_national():
    paths = [ROOT/'covid19_dataset.csv', ROOT/'data'/'covid19_dataset.csv']
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        st.error('Missing covid19_dataset.csv. Keep the original CSV in the repository root or data/ folder.')
        st.stop()
    df = pd.read_csv(path)
    required = {'date','cases'}
    if not required.issubset(df.columns):
        st.error(f'Dataset needs {required}; found {list(df.columns)}')
        st.stop()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['cases'] = pd.to_numeric(df['cases'], errors='coerce')
    df = df.dropna(subset=['date','cases']).sort_values('date')
    df = df.groupby('date',as_index=False)['cases'].sum()
    df['cases_ma7'] = df['cases'].rolling(7,min_periods=1).mean()
    return df

@st.cache_data
def generate_regional():
    # Reproduces the synthetic regional simulation from the repository's covid_19.py.
    rng = pd.date_range('2020-01-01','2022-12-31',freq='D')
    t = np.arange(len(rng))
    center = (pd.Timestamp('2021-06-30')-rng[0]).days
    np.random.seed(42)
    rows=[]
    for region in ['North','South','East','West']:
        amp=np.random.randint(10000,20000)
        width=np.random.randint(60,90)
        baseline=np.random.randint(100,300)
        def peak(c,w,a): return a*np.exp(-.5*((t-c)/w)**2)
        series=peak(center,width,amp)+peak(center-180,width*.7,amp*.5)+peak(center+220,width*.8,amp*.6)+np.random.normal(0,amp*.05,len(t))+baseline
        for day, cases in zip(rng,np.maximum(0,series).astype(int)):
            rows.append((day,region,int(cases)))
    return pd.DataFrame(rows,columns=['date','region','cases'])

national = load_national()
regional = generate_regional()

st.sidebar.title('🦠 COVID-19 Analytics')
st.sidebar.caption('Abhishek Shukla • Portfolio project')
page = st.sidebar.radio('Explore', ['Overview','Cases Trends','Regional Analysis','Interactive Map','Data Explorer'])
st.sidebar.divider()
st.sidebar.info('Educational project using **synthetic data**, not real reported infections or current health guidance.')
st.sidebar.link_button('View source on GitHub','https://github.com/abhishek-shukla7/Covid-19-data-analysis')

st.title('COVID-19 Data Analysis Dashboard')
st.caption('2020–2022 | Synthetic dataset | Interactive analytics')

if page in ['Overview','Cases Trends','Data Explorer']:
    lo,hi = national.date.min().date(),national.date.max().date()
    selected = st.sidebar.date_input('National date range',(lo,hi),min_value=lo,max_value=hi)
    if not isinstance(selected,tuple) or len(selected)!=2:
        st.info('Select both start and end dates.')
        st.stop()
    start,end=selected
    d=national[national.date.dt.date.between(start,end)].copy()
    if d.empty:
        st.warning('No records in this range.')
        st.stop()
    # Recalculate within the chosen interval, so moving averages reflect the filter.
    d['ma7_filtered']=d.cases.rolling(7,min_periods=1).mean()

if page == 'Overview':
    peak=d.loc[d.cases.idxmax()]
    smooth_peak=d.loc[d.ma7_filtered.idxmax()]
    a,b,c,dcol=st.columns(4)
    a.metric('Total synthetic cases',f'{d.cases.sum():,.0f}')
    b.metric('Peak daily cases',f'{peak.cases:,.0f}')
    c.metric('Peak daily date',peak.date.strftime('%d %b %Y'))
    dcol.metric('Peak 7-day average',f'{smooth_peak.ma7_filtered:,.0f}')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.date,y=d.cases,name='Daily cases',line=dict(color='#61a9ff',width=1),opacity=.4))
    fig.add_trace(go.Scatter(x=d.date,y=d.ma7_filtered,name='7-day moving average',line=dict(color='#1fe0c5',width=3)))
    fig.update_layout(title='Synthetic daily cases and 7-day moving average',template='plotly_dark',height=440,hovermode='x unified',paper_bgcolor='#081827',plot_bgcolor='#102b45')
    st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        monthly=d.set_index('date').resample('MS').cases.sum().reset_index()
        fig=px.bar(monthly,x='date',y='cases',title='Cases by month',color='cases',color_continuous_scale='Blues',template='plotly_dark')
        st.plotly_chart(fig,use_container_width=True)
    with right:
        yearly=d.assign(year=d.date.dt.year).groupby('year',as_index=False).cases.sum()
        fig=px.bar(yearly,x='year',y='cases',title='Cases by year (selected dates)',text_auto=',',template='plotly_dark')
        fig.update_xaxes(type='category')
        st.plotly_chart(fig,use_container_width=True)

elif page == 'Cases Trends':
    st.subheader('National trend analysis')
    interval=st.radio('Aggregation',['Daily','Weekly','Monthly'],horizontal=True)
    if interval=='Daily':
        plotted=d[['date','cases']].copy()
    else:
        rule='W' if interval=='Weekly' else 'MS'
        plotted=d.set_index('date').resample(rule).cases.sum().reset_index()
    fig=px.line(plotted,x='date',y='cases',title=f'{interval} synthetic case totals',template='plotly_dark')
    st.plotly_chart(fig,use_container_width=True)
    fig=px.line(d,x='date',y='ma7_filtered',title='7-day moving average',template='plotly_dark')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Moving average is recalculated from the first date in the selected range.')

elif page in ['Regional Analysis','Interactive Map']:
    st.warning('Regional views use the separate, reproducible synthetic simulation in covid_19.py. They are NOT a breakdown of the published national CSV; totals may differ.')
    lo,hi=regional.date.min().date(),regional.date.max().date()
    selection=st.sidebar.date_input('Regional date range',(lo,hi),min_value=lo,max_value=hi,key='regional_dates')
    if not isinstance(selection,tuple) or len(selection)!=2:
        st.stop()
    start,end=selection
    regions=st.sidebar.multiselect('Regions',['North','South','East','West'],default=['North','South','East','West'])
    r=regional[regional.date.dt.date.between(start,end)&regional.region.isin(regions)].copy()
    if r.empty:
        st.info('Choose at least one region and a date range with observations.')
        st.stop()
    summary=r.groupby('region',as_index=False).agg(total_cases=('cases','sum'),peak_daily=('cases','max'),average_daily=('cases','mean'))
    if page=='Regional Analysis':
        st.subheader('Regional synthetic case comparisons')
        a,b=st.columns(2)
        with a:
            st.plotly_chart(px.bar(summary,x='region',y='total_cases',color='region',title='Total synthetic cases by region',template='plotly_dark'),use_container_width=True)
        with b:
            st.plotly_chart(px.bar(summary,x='region',y='peak_daily',color='region',title='Peak daily cases by region',template='plotly_dark'),use_container_width=True)
        trend=r.sort_values('date').copy()
        trend['ma7']=trend.groupby('region').cases.transform(lambda x:x.rolling(7,min_periods=1).mean())
        st.plotly_chart(px.line(trend,x='date',y='ma7',color='region',title='Regional 7-day moving averages',template='plotly_dark'),use_container_width=True)
        st.dataframe(summary,use_container_width=True,hide_index=True)
    else:
        st.subheader('Interactive regional map')
        coords={'North':(28.7041,77.1025),'South':(12.9716,77.5946),'East':(22.5726,88.3639),'West':(19.0760,72.8777)}
        summary['lat']=summary.region.map(lambda v:coords[v][0])
        summary['lon']=summary.region.map(lambda v:coords[v][1])
        st.caption('Markers use representative cities (Delhi, Bengaluru, Kolkata, Mumbai); they are not region boundaries or real epidemiological locations.')
        fig=px.scatter_map(summary,lat='lat',lon='lon',size='peak_daily',color='region',hover_name='region',hover_data={'total_cases':':,','peak_daily':':,','lat':False,'lon':False},zoom=3.6,center={'lat':22,'lon':80},height=570,title='Peak simulated cases by region',map_style='open-street-map')
        fig.update_layout(margin=dict(l=0,r=0,t=45,b=0))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(summary.drop(columns=['lat','lon']),hide_index=True,use_container_width=True)

else:
    st.subheader('Explore the published synthetic dataset')
    st.dataframe(d.rename(columns={'ma7_filtered':'filtered_ma7'}),use_container_width=True,hide_index=True,height=500)
    st.download_button('Download filtered CSV',d.to_csv(index=False).encode('utf-8'),'covid19_filtered.csv','text/csv')
    st.markdown('**Dataset fields:** `date`, `cases`, and a calculated 7-day moving average. No real-world country or state identifiers are present in this CSV.')

st.divider()
st.caption('Portfolio demonstration • Synthetic data • Not suitable for public-health decisions')
