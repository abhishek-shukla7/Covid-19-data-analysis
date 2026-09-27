"""COVID-19 synthetic-data portfolio dashboard. Run: streamlit run streamlit_app.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='COVID-19 | Data Analytics Portfolio', page_icon='🦠', layout='wide')
ROOT = Path(__file__).resolve().parent
COLORS = {'North':'#50B8FF','South':'#2ED3A7','East':'#FFC857','West':'#FF7891'}
COORDS = {'North':(28.7041,77.1025),'South':(12.9716,77.5946),'East':(22.5726,88.3639),'West':(19.0760,72.8777)}
st.markdown('''<style>.stApp{background:#071522;color:#ecf5ff}[data-testid="stSidebar"]{background:#0d2033}.block-container{padding-top:1.5rem}.hero{background:linear-gradient(95deg,#102f4b,#16415d);padding:22px;border:1px solid #317ca9;border-radius:16px;margin-bottom:15px}.hero h1{font-size:2rem;margin:0}.note{font-size:.85rem;color:#a7c2d6}</style>''',unsafe_allow_html=True)

@st.cache_data
def load_daily():
    candidates = [ROOT/'covid19_dataset.csv',ROOT/'data'/'covid19_dataset.csv']
    path = next((p for p in candidates if p.is_file()),None)
    if path is None:
        st.error('Missing covid19_dataset.csv. Keep your existing CSV in the repository root or data/ folder.')
        st.stop()
    df = pd.read_csv(path)
    if not {'date','cases'}.issubset(df.columns):
        st.error('The CSV must contain date and cases columns.')
        st.stop()
    df['date'] = pd.to_datetime(df['date'],errors='coerce')
    df['cases'] = pd.to_numeric(df['cases'],errors='coerce')
    df = df.dropna(subset=['date','cases']).sort_values('date').drop_duplicates('date')
    df['cases_ma7'] = df['cases'].rolling(7,min_periods=1).mean()
    return df

@st.cache_data
def generate_regional():
    """Reproduce synthetic regional generator in original covid_19.py (seed=42)."""
    rng = pd.date_range('2020-01-01','2022-12-31',freq='D')
    t = np.arange(len(rng))
    center = (pd.Timestamp('2021-06-30')-rng[0]).days
    np.random.seed(42)
    parts=[]
    for region in ['North','South','East','West']:
        amp=np.random.randint(10000,20000)
        width=np.random.randint(60,90)
        baseline=np.random.randint(100,300)
        def peak(x,c,w,a): return a*np.exp(-.5*((x-c)/w)**2)
        values=(peak(t,center,width,amp)+peak(t,center-180,width*.7,amp*.5)
                +peak(t,center+220,width*.8,amp*.6)
                +np.random.normal(0,amp*.05,len(t))+baseline)
        parts.append(pd.DataFrame({'date':rng,'region':region,'cases':np.maximum(0,values).astype(int)}))
    return pd.concat(parts,ignore_index=True)

def chart(fig,height=410):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(15,38,58,.4)',height=height,margin=dict(l=12,r=12,t=55,b=20),font=dict(color='#e9f4ff'))
    st.plotly_chart(fig,use_container_width=True)

def date_filter(df,key):
    start,end=df.date.min().date(),df.date.max().date()
    value=st.sidebar.date_input('Date range',(start,end),min_value=start,max_value=end,key=key)
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        st.info('Choose a start and end date in the sidebar.')
        st.stop()
    return df[df.date.dt.date.between(value[0],value[1])].copy()

def header(title,subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><div class="note">{subtitle}</div></div>',unsafe_allow_html=True)

daily=load_daily()
regional=generate_regional()
st.sidebar.title('🦠 COVID-19 Analytics')
st.sidebar.caption('Abhishek Shukla | Portfolio project')
page=st.sidebar.radio('Navigate',['Overview','Cases Trend','Regional Analysis','Regional Map','Data Explorer'])
st.sidebar.divider()
st.sidebar.warning('Educational project using synthetic data (2020–2022). Not official public-health statistics.')

if page=='Overview':
    header('COVID-19 | Analytics Overview','Historical synthetic daily-case dataset • 2020–2022')
    d=date_filter(daily,'overview_dates')
    if d.empty: st.warning('No records in this period.');st.stop()
    peak=d.loc[d.cases.idxmax()]
    smoothed=d.cases.rolling(7,min_periods=1).mean()
    peak7=d.iloc[smoothed.to_numpy().argmax()]
    a,b,c,e=st.columns(4)
    a.metric('Cases in selected period',f'{d.cases.sum():,.0f}')
    b.metric('Peak daily cases',f'{peak.cases:,.0f}')
    c.metric('Peak daily date',peak.date.strftime('%d %b %Y'))
    e.metric('Peak 7-day average',f'{smoothed.max():,.0f}')
    fig=go.Figure()
    fig.add_trace(go.Bar(x=d.date,y=d.cases,name='Daily cases',marker_color='#4386b9',opacity=.48))
    fig.add_trace(go.Scatter(x=d.date,y=smoothed,name='7-day moving average',line=dict(color='#42e4c1',width=3)))
    fig.update_layout(title='Daily cases and rolling 7-day trend',hovermode='x unified')
    chart(fig,450)
    l,r=st.columns(2)
    with l:
        annual=d.assign(year=d.date.dt.year).groupby('year',as_index=False).cases.sum()
        chart(px.bar(annual,x='year',y='cases',title='Cases by year',text_auto=',',color_discrete_sequence=['#46c3df']),340)
    with r:
        month=d.assign(month=d.date.dt.to_period('M').dt.to_timestamp()).groupby('month',as_index=False).cases.sum()
        chart(px.area(month,x='month',y='cases',title='Monthly cases',color_discrete_sequence=['#40cba9']),340)
    st.caption('The Overview uses the existing repository CSV; the regional generator is shown separately.')

elif page=='Cases Trend':
    header('Cases Trend','Explore the original synthetic daily-case CSV and rolling averages')
    d=date_filter(daily,'trend_dates')
    window=st.sidebar.slider('Moving average (days)',3,30,7)
    d['moving_avg']=d.cases.rolling(window,min_periods=1).mean()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d.date,y=d.cases,name='Daily cases',line=dict(color='#6da6d1',width=1)))
    fig.add_trace(go.Scatter(x=d.date,y=d.moving_avg,name=f'{window}-day average',line=dict(color='#31e6ba',width=3)))
    if not d.empty:
        p=d.loc[d.cases.idxmax()]
        fig.add_annotation(x=p.date,y=p.cases,text='Peak',showarrow=True,arrowhead=2)
    fig.update_layout(title='Daily case trend',hovermode='x unified')
    chart(fig,520)
    d['year']=d.date.dt.year
    d['month']=d.date.dt.month
    heat=d.pivot_table(index='year',columns='month',values='cases',aggfunc='sum')
    chart(px.imshow(heat,labels={'x':'Month','y':'Year','color':'Cases'},title='Monthly case heatmap',aspect='auto',color_continuous_scale='Blues'),330)

elif page=='Regional Analysis':
    header('Regional Analysis','Recreated using the original Python script’s seeded synthetic regional generator')
    d=date_filter(regional,'region_dates')
    regions=st.sidebar.multiselect('Regions',list(COORDS),default=list(COORDS))
    d=d[d.region.isin(regions)]
    if d.empty: st.info('Select at least one region.');st.stop()
    total=d.groupby('region',as_index=False).cases.sum().sort_values('cases',ascending=False)
    chart(px.bar(total,x='region',y='cases',color='region',color_discrete_map=COLORS,title='Regional case totals',text_auto=','),400)
    frames=[]
    for region,group in d.groupby('region'):
        group=group.sort_values('date').copy()
        group['ma7']=group.cases.rolling(7,min_periods=1).mean()
        frames.append(group)
    roll=pd.concat(frames)
    chart(px.line(roll,x='date',y='ma7',color='region',color_discrete_map=COLORS,title='Regional 7-day moving averages'),470)
    peaks=d.loc[d.groupby('region').cases.idxmax(),['region','date','cases']].sort_values('cases',ascending=False)
    st.subheader('Peak daily cases by region')
    st.dataframe(peaks.rename(columns={'date':'Peak date','cases':'Peak cases','region':'Region'}),hide_index=True,use_container_width=True)

elif page=='Regional Map':
    header('Regional Map','Representative city markers used by the original project — not geographic case attribution')
    d=date_filter(regional,'map_dates')
    peaks=d.groupby('region',as_index=False).cases.max().rename(columns={'cases':'peak_cases'})
    peaks['lat']=peaks.region.map(lambda x:COORDS[x][0])
    peaks['lon']=peaks.region.map(lambda x:COORDS[x][1])
    peaks['marker_city']=peaks.region.map({'North':'Delhi','South':'Bengaluru','East':'Kolkata','West':'Mumbai'})
    fig=go.Figure(go.Scattergeo(lat=peaks.lat,lon=peaks.lon,mode='markers+text',text=peaks.region,
        textposition='top center',customdata=peaks[['marker_city','peak_cases']].to_numpy(),
        hovertemplate='%{text} region<br>Marker: %{customdata[0]}<br>Peak synthetic daily cases: %{customdata[1]:,.0f}<extra></extra>',
        marker=dict(size=19,color='#30d7bd',line=dict(color='#eafcff',width=1))))
    fig.update_geos(scope='asia',projection_type='mercator',showland=True,landcolor='#17364b',showocean=True,oceancolor='#071522',
                    showcountries=True,countrycolor='#567489',lataxis_range=[5,37],lonaxis_range=[66,98])
    fig.update_layout(title='Regional peak-case markers')
    chart(fig,540)
    st.dataframe(peaks[['region','marker_city','peak_cases']].rename(columns={'region':'Region','marker_city':'Representative city','peak_cases':'Peak daily cases'}),hide_index=True,use_container_width=True)
    st.info('Markers represent regions using city coordinates from the original script. They are not individual city-level case counts.')

else:
    header('Data Explorer','Filter, inspect, and download the existing dataset or generated regional series')
    source=st.radio('Data source',['Existing daily CSV','Generated regional data'],horizontal=True)
    df=daily if source=='Existing daily CSV' else regional
    d=date_filter(df,'explorer_dates')
    if source=='Generated regional data':
        selected=st.multiselect('Regions',list(COORDS),default=list(COORDS))
        d=d[d.region.isin(selected)]
    st.metric('Records',f'{len(d):,}')
    st.dataframe(d,use_container_width=True,hide_index=True,height=440)
    st.download_button('Download filtered CSV',d.to_csv(index=False).encode('utf-8'),file_name='covid_filtered.csv',mime='text/csv')
