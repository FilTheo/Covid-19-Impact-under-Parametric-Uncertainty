
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from pathlib import Path


import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error


#Building the dashboard
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

#Add a function to return the day of interventation
#Preprocess the dataset a bit
cases = pd.read_csv('D:/Uni_Stuff/complex data project/ts/train.csv')
pop = pd.read_csv('D:/Uni_Stuff/complex data project/ts/population_data.csv')
extra = pd.read_csv('D:/Uni_Stuff/complex data project/ts/covid19countryinfo.csv')

#Preprocessing the demographics dataset
cases = cases.fillna('to_keep')
cases = cases[cases['Province_State'] == 'to_keep']
df = pd.merge(cases , pop , left_on = 'Country_Region' , right_on = 'Name', how = 'inner')
df = df[['Id','Name','Date','ConfirmedCases','Fatalities','Population']]
df = df.drop_duplicates('Id')
cols1 = ['region', 'country',  'tests',
       'testpop', 'density', 'medianage', 'urbanpop', 'quarantine', 'schools',
       'publicplace', 'gatheringlimit', 'gathering', 'nonessential',
       'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64',
       'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019',
       'healthexp', 'healthperpop', 'fertility', 'avgtemp', 'avghumidity']


#Removing some columns regarding cases 
extra = extra[cols1]
cols2 = ['region', 'country','density', 'quarantine', 'schools','nonessential', 'hospibed', 'lung' ]
extra = extra[cols2]
extra['region'] = extra.fillna('to_keep')
extra = extra[extra['region']=='to_keep']
country_df = pd.merge(df, extra, left_on = 'Name' , right_on = 'country', how='inner').drop_duplicates('Id')



### Building the equations for the forecasting  model
#Previous model was extremely slow!
#I have to fix the parameters a bit and define a new simpler model!

#Susceptible: S = -beta * Infected * Susceptible
def dS_dt(beta , I , S):
     return(- beta * I * S)

#Exposed : I = beta* Infected * Exposed/incubation_days
def dE_dt(beta, I , E , S, inc_days):
    #print(beta)
    return(beta * I * S - E/inc_days)

#Infected : I = Exposed / incubation_days - I/ infectious days
def dI_dt(E , inc_days, I , inf_days):
    return(E/inc_days - I/inf_days)

#Hospitalized: Number of people needing to go to a hospital:
#Equation from https://covid19-scenarios.org/
def dH_dt(I , C , H , inf_days , d_hosp , d_crit , prob_assymp , prob_crit_fatal):
    # (If they are not assymptomatics)*(people who are currently infecting) + (prob of those who are not to die)*those critical
    #Divide by the days they are critical - those who are hospitlized
    return((1 - prob_assymp)*(I/inf_days) + ((1 - prob_crit_fatal)*C/d_crit)-(H/d_hosp))

#Critical condition: Those who are in critical condition
#Equation from https://covid19-scenarios.org/
def dC_dt(H , C , d_hosp , d_crit , prob_severe_critical):
    #Prob of those who are in hospital to get severe - those who are already severe
    return( (prob_severe_critical* H/d_hosp ) - (C/d_crit))

#Recovered
def dR_dt(I , H , inf_days , d_hosp , prob_assymp , prob_crit_fatal):
    #Those who are assymptomatic and recover + those who are fatal but make it out
    return( (prob_assymp * I/inf_days) + (1 - prob_crit_fatal)*(H/d_hosp))
    
#Assumption: Assymptomatics dont get severe and to die you have to go to the hospital and get critical
#Deaths
def dD_dt(C , d_crit , prob_crit_fatal ):
    #those who are critical multiplied by the prob
    return ( prob_crit_fatal * C/d_crit)


#Model:

#Defining the model:
def model(t , y , R_t , inc_days , inf_days , d_hosp , d_crit , prob_assymp ,  prob_severe_critical , 
         prob_crit_fatal):
    """
    @Parameters description:
    t : time step
    y : previous values of equations
    R_t : reproduction number
    inc_days : incubation days
    inf_days : infectious days
    d_hosp : days a patient spent on hospital either recoviering or going critical
    d_crit : days a patient is in critical state til recover or die
    prob_assymp : probability of mild or assymptomatic.(Not hospital needed)
    prob_severe_critical : probability going to critical state(while on hospital)
    prob_crit_fatal : probability going from critical to fatal
      =b : testing -> attack rate
    c: contracts
   """
    #I will focus on this later

    #Differencial equations past values
    S , E , I , H , C , R , D = y
    #Will use the hill decaying to check how it operates:
    if callable(R_t):
        reprod = R_t(t)
    else:
        reprod = R_t
    #print(reprod,inf_days)   
    beta = reprod/inf_days 
    #print(reprod)
    #print(beta,inf_days)
    #print()
    #beta = R_t / (inf_days *(1-(1-b)**(c*(I+E+H+C+R+D)/N)))
    
    New_S = dS_dt(beta , I , S)
    New_E = dE_dt(beta , I , E , S , inc_days)
    New_I = dI_dt(E , inc_days , I , inf_days)
    New_H = dH_dt(I , C , H , inf_days , d_hosp , d_crit , prob_assymp , prob_crit_fatal)
    New_C = dC_dt(H , C , d_hosp , d_crit , prob_severe_critical)
    New_R = dR_dt(I , H , inf_days , d_hosp , prob_assymp , prob_crit_fatal)
    New_D = dD_dt(C, d_crit , prob_crit_fatal)
    
    to_return = [New_S , New_E , New_I , New_H , New_C , New_R , New_D]
    return(to_return)
    
   

#Identifies the date a country applied extra measures
#This function is used mainly for calibration
def interventaion_day(country_df):
    country_df['Date'] = pd.to_datetime(country_df['Date'])
    country_df['quarantine'] = pd.to_datetime(country_df['quarantine'])
    int_days = country_df.iloc[0]['quarantine'] - country_df.iloc[0]['Date'] 
    int_days = int_days.days
    return(int_days)

#Extracts initial information for a country
def get_country_dash(country):
  #Getting the country
    test = country_df[country_df['Name'] == country]
    #Getthing the data from which the first case was appeared
    test = test[test['ConfirmedCases'] > 0]
    #Assumption that population remains the same
    population = test['Population'].values[0]
    #Initial confirmed cases:
    infected = test['ConfirmedCases'].values[0]
    return (population , infected) 

#Functions for Rt

#Gets the new Rt depending on the extra parameters
def new_Rt(b,c,a,Rt):
    """
    @params
    b : transimition rate, normal value 0.45(or 0.5), masks shops 0.3 , masks closed spaces 0.2 , masks everywhere 0.1-0.05
    c : contracts per person : normal value : 15, work from home 10 , closed bars 8 , closed restaurants 7, lockdown 5, strict lockdown :3
    a : goverments intervetion : normal value 0.8 , recommendations light : 0.70-0.75, strict 0.65 , fees everywhere 0.5
    The values for every measure should be reconsider
    """
    return (Rt - (1-b)**(c*a)*Rt)


#Lineary decaing RT
def linear_decay(Rold , Rnew , total_steps , step): #t stands for steps of decaying 
    Rt = Rold - ((Rold - Rnew)/total_steps)*step
    return(Rt)

#The function for calculating Rt on each time step
def My_Rt(t):
    if (t > intervention_date) and (t <= intervention_date + intervention_duration):
      #Rt_new = new_Rt(b,c,a,R_0) 
          #If t has not reached the half time of intervention duration for the measures to full applied
        if (t < (intervention_date + intervention_duration/2)):
              #Applying the decaying function
              #This is the decaying step, will decay from step = 1 until step = total_steps
              #For example if intervention is made on step 5 and t = 6, i have step = t - intervention = 1
            step = t - intervention_date
              #until half the interventaion duration for measures to be applied
            Rt = linear_decay(R_0 , Rt_new , intervention_duration/2, step)
            return Rt
          #If time is over half the intervention duration then rt gets the new desired value
        elif (t >= intervention_date + intervention_duration/2):
            return Rt_new
      #If measures have been completed
    elif t > (intervention_date + intervention_duration):
          #still unsure about how to approach this when measures have passed!
          #One idea is to get 0.5*R0 so it is cut is half but maybe not so accurate here
          #Another idea is to give the new Rt value
          #Finaly I can add a increasing function which will slowly return the value back to R0!!
          
          #best idea is the increasing function but is pretty weird to be applied
          #For now I am giving the new Rt
        return Rt_new
      #Finally if the interventaion day have not yet been reached returns R0
    else:
        return R_0


#Building the dashboard!


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#Initializing
app = dash.Dash(external_stylesheets = [dbc.themes.CYBORG])



#Building the Layout
app.layout = dbc.Container([
    #Title 
    dbc.Row([
        dbc.Col([
                html.H1("Forecasting under Parametric Uncertainty Dashboard", 
                                style={'color': '#189de4', 'fontSize': 25 ,'textAlign': 'center', "width":"100%"}
                        )], 
                        width = 12
                ) 
            ]),
    
    #Initialize graph: Pick a Country and what to visualize
    dbc.Row([
        dbc.Col(
                 width = {'size': 2}
                ),
        dbc.Col([
                dcc.Dropdown(
                              id = 'country',
                              options = [
                                         {'label': 'Greece', 'value': 'Greece'},
                                         {'label': 'UK', 'value': 'United Kingdom'},
                                         {'label': 'Sweden', 'value': 'Sweden'},
                                        ],
                               #value = 'manual', 
                             placeholder = "Pick a country",
                              style={'width': 200 ,"margin-left": "10px",'color': '#212121'}
                             )
          
          
                  ], width = {'size' : 3.5}
                  ),
        dbc.Col([
                  html.Label('Forecast horizon',
                            style = {'color': '#89cff0', "margin-left": "35px","margin-right": "5px" }),
                  dcc.Input(
                                id = 'days',
                                placeholder = 200,
                                type = 'number',
                                value = 200 
                                #Style: width controls the width of the box
                                #       margin left and right control the empty space from the left and from the right
                                , style={'width': 50 ,"margin-left": "35px","margin-right": "5px" } 
                            )
                ] , width = {'size': 2}
               ),
        
        dbc.Col([
                 html.Label('Graph Options', style = {'color': '#89cff0'}),
                 dcc.Checklist(
                                 options = [
                                            {'label': 'Exposed', 'value': 'E'},
                                            {'label': u'Infectious', 'value': 'I'},
                                            {'label': u'Hospitalized', 'value': 'H'},
                                            {'label': u'Critical', 'value': 'C'},
                                            {'label': 'Deceased', 'value': 'D'}
                                            ],
                                  value=['E', 'I'], 
                                  inputStyle = { "margin-right": "5px","margin-left": "5px"},
                                  labelStyle = {'display': 'inline-block', 'color': '#dddddd'}
                                  )
        
                ], width = {'size' : 5}
                )

            ]),
    
    #Pandemic Dynamics & Graph
    dbc.Row([   
        dbc.Col([   
                    html.H3('Pandemic Dynamics' , style = {'color': '#189de4', 'fontSize': 20}),

                    html.Label('Incubation Days', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'inc_days',
                               min = 4, max = 21, value = 8, step = 1, 
                               marks = {
                                        4 : {"label" : "4"},
                                        10 : {"label" : "10"},
                                        14 : {"label" : "14"},
                                        18 : {"label" : "18"},
                                        21 : {"label" : "21"},
                                       }),
                    html.Br(),
            
                    html.Label('Infectious Days', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'inf_days',
                               min = 1, max = 10, value = 4, step = 1,
                               marks = {
                                        1 : {"label" : "1"},
                                        3 : {"label" : "3"},
                                        5 : {"label" : "5"},
                                        8 : {"label" : "8"},
                                        10 : {"label" : "10"},
                                       }),
                    html.Br(),
                    
                    html.Label('Hospital Days', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'hosp_days',
                               min = 1, max = 15, value = 4, step = 1,
                               marks = {
                                        1 : {"label" : "1"},
                                        4 : {"label" : "4"},
                                        7 : {"label" : "7"},
                                        10 : {"label" : "10"},
                                        12 : {"label" : "12"},
                                        15 : {"label" : "15"},
                                       }),
                    html.Br(),
                    
                    html.Label('Critical Days', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'crit_days',
                               min = 1, max = 21, value = 7, step = 1,
                               marks = {
                                        1 : {"label" : "1"},
                                        4 : {"label" : "4"},
                                        7 : {"label" : "7"},
                                        12 : {"label" : "14"},
                                        18 : {"label" : "18"},
                                        21 : {"label" : "21"},
                                       }),
                    html.Br(),
                    
                    html.Label('Proportion of Asymptomatic', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'prop_asympt',
                               min = 0, max = 1, value = 0.6 ,step = 0.1,
                               marks = {
                                        0 : {"label" : "0"},
                                        0.4 : {"label" : "0.4"},
                                        0.8 : {"label" : "0.8"},
                                        1 : {"label" : "1"},
                                       }),
                    html.Br(),
                    
                    html.Label('Frequency of Critical Symptoms', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'freq_crit',
                               min = 0, max = 1, value = 0.3 ,step = 0.1,
                               marks = {
                                        0 : {"label" : "0"},
                                        0.4 : {"label" : "0.4"},
                                        0.8 : {"label" : "0.8"},
                                        1 : {"label" : "1"},
                                       }),
                    html.Br(),
                    
                    html.Label('Mortality Rate after Critical Symptoms', style = {'color': '#89cff0'}),
                    dcc.Slider(id = 'mort_crit',
                               min = 0, max = 1, value = 0.1 ,step = 0.1,
                               marks = {
                                        0 : {"label" : "0"},
                                        0.4 : {"label" : "0.4"},
                                        0.8 : {"label" : "0.8"},
                                        1 : {"label" : "1"},
                                       }),
                    html.Br(),
                    
                    #Ending sliders here and adding the initial reproduction number
                    
                    #Some Empty space: Each Br is an empty line
                    #html.Br(),
                    #html.Br(),
        ], width = {'size': 3} #giving size 5 to the full column , From the initial 12 on the title I have 7 columns more 
        ),
        dbc.Col([
                 #Right hand side graph
                 dcc.Graph( 
                            id = 'main_graph' 
                           # ,figure = fig 
                 )
                 ], width = {'size': 9}
                 )
           
    ]),

    #Third row : The actions
    dbc.Row([
        #Empty space
        dbc.Col(
            width = {'size': 3}
                ),        
        #Second column is the measures
        dbc.Col([ 
                  html.Label('Intervention date', style = {'color': '#89cff0'}),
                  dcc.Input(
                                id = 'interv_day',
                                placeholder = 35,
                                type = 'number',
                                value = 35 ,
                                #Style: width controls the width of the box
                                #       margin left and right control the empty space from the left and from the right
                                style={'width': 45 ,"margin-left": "5px","margin-right": "35px" } 
                            )
                ] , width = {'size':1.8}
               ),
        
        dbc.Col([
                 html.Label('Intervention duration', style = {'color': '#89cff0'}),
                 dcc.Input(
                            id = 'interv_durat',
                            placeholder = 11,
                            type = 'number',
                            value = 11 ,
                            #Style controls the width of the box
                            style={'width': 45,"margin-left": "7px","margin-right": "35px" } 
                           ),
                 html.Label('Recommendations', style = {'color': '#89cff0'}),
            
                ], width = {'size': 2.8}
                ),
        
        #The recommnedations dropdown
        dbc.Col([
                dcc.Dropdown(
                              id='rec_drop',
                              options = [
                                         {'label': 'Total Lockdown', 'value': 'tot'},
                                         {'label': 'Obligatory Masks & Work from Home', 'value': 'Maskhome'},
                                         {'label': 'Obligatory Masks, Closed Bars, Work from Home', 'value': 'MaskBarHome'},
                                         {'label': 'Recommendations for Masks and avoid Gatherings', 'value': 'rec'},
                                         {'label': 'Manual Setting', 'value': 'manual'}
                                        ],
                               value = 'manual', 
                              style={'width': 300 ,"margin-left": "10px",'color': '#212121'}
                            ) 
                ], width = {'size': 3.96}
                )
                 
            ]),
    
    
     #An empty row to add some gap between the two rows
    dbc.Row([
         html.Br()
             ]),
    
    
    
    #Last row: Manual measures
    dbc.Row([
        
        #Empty column: Still do not what to add here -> maybe some manual calibration for a country

        dbc.Col([
                html.Label('Initial reproduction number R0', style = {'color': '#89cff0'}),
                dcc.Slider(id = 'initial_r0',
                           min = 0.1, max = 5, value = 0.8, step = 0.1,
                           marks = {
                                    0.1 : {"label" : "0.1"},
                                    0.5 : {"label" : "0.5"},
                                    1 : {"label" : "1"},
                                    1.5 : {"label" : "1.5"},
                                    2 : {"label" : "2"},
                                    2.5 : {"label" : "2.5"},
                                    3 : {"label" : "3"},
                                    3.5 : {"label" : "3.5"},
                                    4 : {"label" : "4"},
                                    4.5 : {"label" : "4.5"},
                                    5 : {"label" : "5"},
                                   }),
                ], width = {'size': 3}
                ),
        #Second column is the measures
     dbc.Col([ 
                  html.Label('Transmition Rate', style = {'color': '#89cff0'}),
                  dcc.Input(
                                id = 'transm_rate',
                                placeholder = 0.2,
                                type = 'number',
                                value = 0.2 ,
                                #Style: width controls the width of the box
                                #       margin left and right control the empty space from the left and from the right
                                style={'width': 45 ,"margin-left": "30px","margin-right": "35px"} 
                            ),
                  html.Label('Normal Setting: 0.45', style = {'color': '#dddddd'}),
                ], width = {'size': 2}, align = 'top' 
               ),
        dbc.Col([ 
                  html.Label('Contacts per Day', style = {'color': '#89cff0'}),
                  dcc.Input(
                                id = 'cont_day',
                                placeholder = 15,
                                type = 'number',
                                value = 15 ,
                                #Style: width controls the width of the box
                                #       margin left and right control the empty space from the left and from the right
                                style={'width': 45 ,"margin-left": "30px","margin-right": "35px"},

                            ),
                  html.Label('Normal Setting: 15', style = {'color': '#dddddd'}),
                ], width = {'size': 2}, align = 'center'
               ),
        dbc.Col([ 
                  html.Label('Guideliness Strictness', style = {'color': '#89cff0'}),
                  dcc.Input(
                                id = 'strict',
                                placeholder = 0.9,
                                type = 'number',
                                value = 0.9 ,
                                #Style: width controls the width of the box
                                #       margin left and right control the empty space from the left and from the right
                                style={'width': 45 ,"margin-left": "30px","margin-right": "35px"}
                            ),
                  html.Label('Normal Setting: 0.9', style = {'color': '#dddddd'}),
                ], width = {'size': 2}, align = 'center'
               )

        
        
    ])
    
])


#Functions used for the callbacks to update the graph

#Initializing the values from the dashboard
def initialize_vals(inc_days, inf_days, hosp_days, crit_days, prop_asympt, freq_crit, mort_crit, initial_r0, 
                            interv_day, interv_durat, transm_rate, cont_day, strict):
    incubation_days = float(inc_days)
    infection_days = float(inf_days)
    hospital_days = float(hosp_days)
    Critical_days = float(crit_days)
    Assymptomatics = float(prop_asympt)
    Critical_Symptoms = float(freq_crit)
    Mortality_After_Critical = float(mort_crit)
    #Intervention_day = float(interv_day)
    #Intervention_duration = float(interv_durat)
    Transmition_rate = float(transm_rate)
    Contracts_Per_day = float(cont_day)
    Measures_Strictness = float(strict)
    #Get the new RT
    Rt_new = new_Rt(Transmition_rate, Contracts_Per_day, Measures_Strictness, float(initial_r0))
    
    intervention_date = float(interv_day)
    intervention_duration = float(interv_durat)
    args = (My_Rt , incubation_days , infection_days , hospital_days , Critical_days ,
            Assymptomatics , Critical_Symptoms , Mortality_After_Critical)
    return args

#Solving the system
def solve_system(args, days, N, n_inf):
    #Initialize population and infections
    initials = [(N - n_inf)/ N, 0, n_inf/N, 0, 0, 0, 0]
    days = int(days)
    #Solving the system of equations
    solution = solve_ivp(model, [0, days], initials, args = args, t_eval = np.arange(days))
    S , E , I , H , C , R , D = solution.y

    #total cases: infected + hospitlized + critical + recovered + dead
    total_cases = I + H + C + R + D 
    total_cases = np.clip(I + E + H + C + R + D, 0 , np.inf)
    #40 is picked to start the graph empiricaly: It will be modified on the final version to be user manipulated
    S = N*S[40:]
    E = E[40:]*N
    I = I[40:]*N
    H = H[40:]*N
    C = C[40:]*N
    R = R[40:]*N
    D = D[40:]*N
    return S, E, I, H, C, R, D

#Building the graph
def build_graph(df):
    fig = go.Figure()
    #For starting: Plot everything and then include the checklists
    fig.add_trace(go.Bar(x = df['Day'], y = df['Exposed'], name = 'Exposed' , marker_line_width=0,width = 0.7))
    fig.add_trace(go.Bar(x = df['Day'], y = df['Infected'], name = 'Infected' , marker_line_width=0,width = 0.7))
    fig.add_trace(go.Bar(x = df['Day'], y = df['Hospitalized'], name = 'Hospitalized' , marker_line_width=0,width = 0.7))
    fig.add_trace(go.Bar(x = df['Day'], y = df['Critical'], name = 'Critical' , marker_line_width=0,width = 0.7))
    fig.add_trace(go.Bar(x = df['Day'], y = df['Deceased'], name = 'Deceased' , marker_line_width=0,width = 0.7))
    fig.update_layout(
        title={
                'text': "Daily Forecasts",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
        xaxis = dict(title = 'Day'),
        xaxis_tickfont_size=14,
        yaxis=dict(
            title = 'Population',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            yanchor="top",
            xanchor="left",
            x = 0.85,
            y = 1,
            font=dict(size=12,
                    color="black"),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2),
        bargap=0.01, # gap between bars of adjacent location coordinates.
        bargroupgap=0.01 # gap between bars of the same location coordinate.
    )
    return fig


#Building the callback
@app.callback(
    Output('main_graph' , 'figure'),
    #Initials
    Input('country', 'value'),
    Input('days','value'),
    #Pandemic dynamics
    Input('inc_days','value'),
    Input('inf_days','value'),
    Input('hosp_days','value'),
    Input('crit_days','value'),
    Input('prop_asympt','value'),
    Input('freq_crit','value'),
    Input('mort_crit','value'),
    #Initial r_0
    Input('initial_r0', 'value'),
    #Intervention dates
    Input('interv_day', 'value'),
    Input('interv_durat', 'value'),
    #Intervention types
    Input('transm_rate','value'),
    Input('cont_day','value'),
    Input('strict','value'),
    Input('rec_drop','value'),
)

def update_graph(country, days, inc_days, inf_days, hosp_days, crit_days, prop_asympt, freq_crit, mort_crit, initial_r0, 
                interv_day, interv_durat, transm_rate, cont_day, strict,  rec_drop = 'manual'):
    
    #Get measures strictness
    
    
    
    #Initializing
    args = initialize_vals(inc_days, inf_days, hosp_days, crit_days, prop_asympt, freq_crit, mort_crit, initial_r0, 
                            interv_day, interv_durat, transm_rate, cont_day, strict)
    #Initialize population 
    count_specific = get_country_dash(country)
    N = count_specific[0]
    n_inf = count_specific[1]
    #Initial infections
    #Solving the system
    solution = solve_system(args, days, N, n_inf)
    S , E , I , H , C , R , D = solution
    
    #Initializing values
    total_days = np.arange(40, days, 1) #Again 40 will be modified
    df = pd.DataFrame([total_days,E,I,H,C,R,D]).T
    df = df.rename(columns = {0 : 'Day', 1 : 'Exposed', 2 : 'Infected', 3 : 'Hospitalized', 4 : 'Critical',
                              5 : 'Recovered' ,6 : 'Deceased'})
    
    #Building the graph:
    fig = build_graph(df)
    
    fig.update_layout(transition_duration = 500)
    
    return fig


#Initializing
if __name__ == '__main__':
    app.run_server(debug = False)
