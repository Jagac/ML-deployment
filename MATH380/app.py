import streamlit as st


               
st.title('Model Gui')


st.write("This is a cool front end for our model. Feel free to change the weights and the app will do all the calculations for you")
st.write("")





w1 = st.sidebar.slider('No health insurance coverage',0.01,1.0)
w2 = st.sidebar.slider('Unemployed',0.01,1.0)
w3 = st.sidebar.slider('No vehicles available', 0.01,1.0)
w4 = st.sidebar.slider('Child', 0.01,1.0)
w5 = st.sidebar.slider('Foreign Born', 0.01,1.0)
w6 = st.sidebar.slider('Speak English less than "very well"',0.01,1.0)
w7 = st.sidebar.slider('Without a broadband Internet subscription',0.01,1.0)
w8 = st.sidebar.slider('Median age (years)',0.01,1.0)
w9 = st.sidebar.slider('Less than $25,000',0.01,1.0)
w10 = st.sidebar.slider('Less than 12th grade',0.01,1.0)


if st.button('Calculate the Weighted Mean'):
    import pandas as pd
    import numpy as np
    df = pd.read_csv("Walworth_County_ACS_Data_2014-2020_Group_1.csv")
    df1=df.fillna(0)
    df1['Less than $25,000'] = df1['Less than $10,000'] + df1['$10,000 to $14,999'] + df1['$15,000 to $24,999']
    df1['Less than 12th grade'] = df1['Less than 9th grade'] + df1['9th to 12th grade, no diploma']
    df1['Without a broadband Internet subscription'] = 1-df1['With a broadband Internet subscription']

    df2 = df1.drop(['Less than $10,000','$10,000 to $14,999', '$15,000 to $24,999','Less than 9th grade',
                '9th to 12th grade, no diploma','Average family size',
               'With a broadband Internet subscription'], axis=1)
    
    df2['Weighted Mean'] = (df2['No health insurance coverage']*w1 + df2['Unemployed']*w2 + df2['No vehicles available']*w3
                        + df2['Child']*w4 + df1['Foreign born']*w5 +df2['Speak English less than "very well"']*w6 + 
                        df2['Without a broadband Internet subscription']*w7 + df2['Median age (years)']*w8 + 
                        df2['Less than $25,000']*w9 + df1['Less than 12th grade']*w10)/(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10)
    from sklearn.preprocessing import StandardScaler
    df2[df2.columns[2:]]=StandardScaler().fit_transform(df2[df2.columns[2:]])

    df2['Weighted Mean'] = (df2['No health insurance coverage']*w1 + df2['Unemployed']*w2 + df2['No vehicles available']*w3
                        + df2['Child']*w4 + df1['Foreign born']*w5 +df2['Speak English less than "very well"']*w6 + 
                        df2['Without a broadband Internet subscription']*w7 + df2['Median age (years)']*w8 + 
                        df2['Less than $25,000']*w9 + df1['Less than 12th grade']*w10)/(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10)
    df3 = df2.sort_values(['Year','Weighted Mean'], ascending = (False, False))
    df3.to_csv('walwortwheightedmean.csv', index = False)

    'DONE'
    st.write(df3[df3['Year']==2020][['Municipality','Year','Weighted Mean']].head(3))
    '---'
    st.write(df3[df3['Year']==2019][['Municipality','Year','Weighted Mean']].head(3))
    '---'
    st.write(df3[df3['Year']==2018][['Municipality','Year','Weighted Mean']].head(3))
    '---'
    st.write(df3[df3['Year']==2017][['Municipality','Year','Weighted Mean']].head(3))
    '---'
    st.write(df3[df3['Year']==2016][['Municipality','Year','Weighted Mean']].head(3))
    '---'
    st.write(df3[df3['Year']==2015][['Municipality','Year','Weighted Mean']].head(3))
    '---'
    st.write(df3[df3['Year']==2014][['Municipality','Year','Weighted Mean']].head(3))



if st.button('Run ARIMA Model'):
    import pandas as pd
    import numpy as np                 
    import statsmodels.api as sm  
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error  
    'DONE'
    df3 = pd.read_csv('walwortwheightedmean.csv')
    df_temp=df3[['Municipality','Year','Weighted Mean']]
    muns=['Bloomfield town', 'Whitewater city', 'Delavan city',
       'Darien town', 'Walworth village', 'Sharon village',
       'Lake Geneva city', 'Bloomfield village', 'Elkhorn city',
       'Walworth town', 'Linn town', 'Williams Bay village',
       'Genoa City village', 'Delavan town', 'Geneva town',
       'Darien village', 'Richmond town', 'Sharon town',
       'Fontana-on-Geneva Lake village', 'Whitewater town',
       'East Troy village', 'Spring Prairie town', 'East Troy town',
       'Troy town', 'La Grange town', 'Lyons town', 'Lafayette town',
       'Sugar Creek town', 'Mukwonago village', 'Burlington city']
    def ARIMA(df):
        x=df.sort_values('Year')['Weighted Mean']
        model = sm.tsa.arima.ARIMA(x.to_numpy(), order=(0,0,0))
        model_fit = model.fit(x)
        output=model_fit.forecast()
        return output
    for x in muns:
        [output]=ARIMA(df_temp[df_temp['Municipality']==x])
        row={'Municipality':x,'Year':2021,'Weighted Mean':output}
        print(row)
        df_temp=df_temp.append(row,ignore_index=True)
    def plot_graph(mun):
        global df_temp
        ddf=df_temp[df_temp['Municipality']==mun].sort_values('Year')
        fig, ax = plt.subplots()
        ax.plot(ddf['Year'], ddf['Weighted Mean'])
        #ax.set_xlabel(ddf['Year'])
        return fig
    for mun in muns:
        st.write(mun)
        st.pyplot(plot_graph(mun))

