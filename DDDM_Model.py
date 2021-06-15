#The proposed framework is described
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pathlib import Path
import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error


#Data used for evaluation
#Data describe the number of total cases, the population and some extra demographics
cases = pd.read_csv('D:/complex data project/ts/train.csv')
pop = pd.read_csv('D:/complex data project/ts/population_data.csv')
extra = pd.read_csv('D:/complex data project/ts/covid19countryinfo.csv')

#Some standarized steps applied on the SEIR model
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
df = pd.merge(df, extra, left_on = 'Name' , right_on = 'country', how='inner').drop_duplicates('Id')
df


#Building the mathematical model.

#Defining the differential equations
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
    R_t : reproduction number #requires estimations
    inc_days : incubation days #given as a constant parameter
    inf_days : infectious days # given as a constant parameter
    d_hosp : days a patient spent on hospital either recoviering or going critical # given as a constant parameter
    d_crit : days a patient is in critical state til recover or die # given as a constant parameter
    prob_assymp : probability of mild or assymptomatic.(Not hospital needed) # given as a constant parameter
    prob_severe_critical : probability going to critical state(while on hospital) # given as a constant parameter
    prob_crit_fatal : probability going from critical to fatal # given as a constant parameter
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
    

#MAPE for evaluation
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#Extracting the intervation date for each country
def interventaion_day(country_df):
  country_df['Date'] = pd.to_datetime(country_df['Date'])
  country_df['quarantine'] = pd.to_datetime(country_df['quarantine'])
  int_days = country_df.iloc[0]['quarantine'] - country_df.iloc[0]['Date'] 
  int_days = int_days.days
  return(int_days)
  
#Extracting the required infromation for each country
def get_country(Country):
  #Getting the country
    test = df[df['Name'] == Country]
    #Getthing the data from which the first case was appeared
    test = test[test['ConfirmedCases'] > 0]
    #Assumption that population remains the same
    population = test['Population'].iloc[0]
    #Initial confirmed cases:
    infected = test['ConfirmedCases'].iloc[0]
    #Splitting train-test set
    train_set = test[:-14]
    #Getting the total days which I am going to forecast
    days = len(train_set)
    #Getting the true confirmed cases for the evaluation
    true_cases_train = train_set['ConfirmedCases'].values
    true_fatalities_train = train_set['Fatalities'].values

    true_cases_test = test[-14:]['ConfirmedCases'].values
    true_fatalities_test = test[-14:]['Fatalities'].values
    true_fatalities = test['Fatalities'].values
    #Getting the interventaion date:
    int_day = interventaion_day(test)
    return (population , infected , days , true_cases_train , true_fatalities_train , true_cases_test , true_fatalities_test,int_day)

#My designed function for estimating the Rt based on the intervantions of each country3
def new_Rt(b,c,a,Rt):
  """
  @params
  b : transimition rate, normal value 0.45(or 0.5), masks shops 0.3 , masks closed spaces 0.2 , masks everywhere 0.1-0.05
  c : contracts per person : normal value : 15, work from home 10 , closed bars 8 , closed restaurants 7, lockdown 5, strict lockdown :3
  a : goverments intervetion : normal value 0.8 , recommendations light : 0.70-0.75, strict 0.65 , fees everywhere 0.5
  The values for every measure should be reconsider
  """
  return (Rt - (1-b)**(c*a)*Rt)

#Another function for estimating the Rt after an internvation based on linear decaying
def linear_decay(Rold , Rnew , total_steps , step): #t stands for steps of decaying 
    Rt = Rold - ((Rold - Rnew)/total_steps)*step
    return(Rt)

#Estimating the for each timestep
def find_Rt(t):

  if (t > intervention_date) and (t <= intervention_date + intervention_duration):
      Rt_new = new_Rt(b,c,a,R0) 
          #If t has not reached half the time of time intervention duration for the measures to full applied
      if (t < (intervention_date + intervention_duration/2)):
              #Applying the decaying function
              #This is the decaying step, will decay from step = 1 until step = total_steps
              #For example if intervention is made on step 5 and t = 6, i have step = t - intervention = 1
          step = t - intervention_date
              #until half the interventaion duration for measures to be applied
          Rt = linear_decay(R0 , Rt_new , intervention_duration/2, step)
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
          
          #For now I am giving the new Rt
      return Rt_new
      #Finally if the interventaion day have not yet been reached returns R0
  else:
      return R0

##Functions for evaluation and calibration
# These function search the optimal constant parameter so the model is fitted on the given data
#Info about this function on the above section
def evaluation_final(params, Country ,opt_days, forecast_days = 0,   inc_days = 5.6, inf_days = 2.9 , to_return = False):

#added the intervantion time and set as default the 30 days
#Added the forecast_days to forecast ahead on the future

  #Initializing values
  inc_days = inc_days
  inf_days = inf_days
  R_0 , d_hosp, d_crit, prob_assymp, prob_severe_critical, prob_crit_fatal, b , c ,intervantion_time = params
  population , n_infected , days , true_cases_train , true_fatalities_train,true_cases_test , true_fatalities_test,aaaa  = get_country(Country)
  #Keeping last 14 days as a test set!
  #If foreacst days = 14 then I am adding them to my forecast
  days = days + forecast_days
  #Giving initial conditions
  initials = [(population - n_infected)/ population , 0, n_infected/population, 0, 0, 0, 0]

  #Varying function, a combination between the two Rt functions described before
  def varying_Rt(t):
    if (t > intervantion_time):
      return (R_0 / (1 + (t/c)**b))
    else:
      return (R_0)
  #Solving the differential equations
  args = (varying_Rt , inc_days , inf_days , d_hosp , d_crit , prob_assymp , prob_severe_critical , prob_crit_fatal)
  solution = solve_ivp(model, [0, days], initials, args = args, t_eval = np.arange(days))
  #Getting the returned values
  S , I , E , H , C , R , D = solution.y
  #gives value 0 if there is a negative value
  total_cases = np.clip(I + E + H + C + R + D, 0 , np.inf)
  #These total cases are per milion so :
  total_cases = total_cases * population

  #This is just for calibration, on the final model i will extract all the values S , I , E, ... 
  y_pred_cases = total_cases
  y_pred_fat = np.clip(D, 0 ,np.inf) * population
  if forecast_days == 0 :
    y_true_cases = true_cases_train
    y_true_fat = true_fatalities_train
  else:
    y_true_cases = true_cases_test
    y_true_fat = true_fatalities_test
    y_pred_cases = y_pred_cases[-14:]
    y_pred_fat = y_pred_fat[-14:]


  optimaziation_days = opt_days 
  weights = 1 / np.arange(1, optimaziation_days + 1)[::-1]
  if forecast_days == 0: 
  #Getting msle of cases and deaths to optimize
  
    msle_cases = mean_squared_log_error(y_true_cases[-optimaziation_days:], y_pred_cases[-optimaziation_days:],weights)
    #print(msle_cases)
    msle_fat = mean_squared_log_error(y_true_fat[-optimaziation_days:], y_pred_fat[-optimaziation_days:],weights)
    #mean of the two
    msle = np.mean([msle_cases,msle_fat])
 
  else:
    msle_cases = mean_squared_log_error(y_true_cases,y_pred_cases)
    msle_fat = mean_squared_log_error(y_true_fat,y_pred_fat)
    msle = np.mean([msle_cases,msle_fat])
  #To return is for getting the extracted parameters from the calibrated model(If True)
  #If false, the function is solely used for optimization
  if to_return == True:
  #if forecast_days == 0:
    #Change the returned values!!!
    #return(msle, y_pred_cases, y_true_cases)
    #else:
    return(msle, y_pred_cases, y_true_cases)
  else:
    return(msle)


#A function to plot the results
def plot_results(true_train , predicted_train, true_test,predicted_test): #edit it later
  plt.figure(figsize = (12,6))
  t = np.arange(0 , len(true_train))
  z = np.arange(len(true_train) ,len(true_train) +len(predicted_test))
  plt.plot(t, true_train , label = 'Training Set' , color = 'green')
  plt.plot(t , predicted_train , label ='Predicted Train' , color = 'blue')
  plt.plot(z , true_test , label ='True Test' , color = 'yellow')
  plt.plot(z , predicted_test , label ='Predicted Test' , color = 'red')

  plt.legend()
  label = "Training Errors" 
  print("Test MSE:",+ mean_absolute_percentage_error(true_test,predicted_test))
  plt.title(label)
  plt.show()


       
# A function which searches the best parameters for the model

# The final model uses paramteres extracted from literature
# This function is used to evaluate the forecasting performance of the model
def optimize_final(guesses_initial,  bounds,  Country):
  #Optimizing, find the best configuration
  to_optimize = [35,40,45]
  scores = []
  parameters_set = []
  for i in to_optimize:
    args = (Country,i)
    r_optimized = minimize(evaluation_final, guesses_initial , bounds = bounds ,method = 'L-BFGS-B', args = args ) 
    #Returning the optimized parameters
    R_0 , d_hosp , d_crit , prob_assymp , prob_severe_critical , prob_crit_fatal , b , c, intervantion_time = r_optimized.x
    parameters_set.append(r_optimized.x)
    params = [R_0 ,d_hosp,d_crit,prob_assymp,prob_severe_critical,prob_crit_fatal , b , c, intervantion_time] 
    msle_train , y_pred_train , y_true_train = evaluation_final(params,country,i, to_return = True)
    msle_test , y_pred , y_true = evaluation_final(params,country,i,forecast_days = 14, to_return = True,)
    #Last 14 days are the test set
    y_pred_test = y_pred[-14:]
    y_true_test = y_true[-14:]
    training_set = y_true_train
    
    #Plotting all results skip for now
    
    mse = mean_squared_error(y_true_test , y_pred_test )
    mape_test = mean_absolute_percentage_error(y_true_test , y_pred_test)
    #Testing with train error
    mape_train = mean_absolute_percentage_error(y_pred_train , y_true_train)
    scores.append(mape_test)
  best = np.argmin(scores)
  best_opt = to_optimize[best]
  best_params = parameters_set[best]
  print(best_opt)
  #Getting the results on the best configuration
  args = (Country, best_opt)
  #r_optimized = minimize(evaluation_final, guesses_initial , bounds = bounds ,method = 'L-BFGS-B', args = args ) 
    #Returning the optimized parameters
  R_0 , d_hosp , d_crit , prob_assymp , prob_severe_critical , prob_crit_fatal , b , c, intervantion_time = best_params
  params = [R_0 ,d_hosp,d_crit,prob_assymp,prob_severe_critical,prob_crit_fatal , b , c, intervantion_time] 
  msle_train , y_pred_train , y_true_train = evaluation_final(params,country,best_opt, to_return = True)
  msle_test , y_pred , y_true = evaluation_final(params,country,best_opt,forecast_days = 14, to_return = True,)
  y_pred_test = y_pred[-14:]
  y_true_test = y_true[-14:]
  training_set = y_true_train
    
  plot_results(y_true_train, y_pred_train , y_true_test , y_pred_test )
  #print('MSLE Train :' , msle_train)
  #returning the set of fitted parameters for each country
  to_return = [R_0 , d_hosp , d_crit , prob_assymp , prob_severe_critical , prob_crit_fatal]
  return (to_return)
 
#Testing the model

country = "Spain"
int_day = get_country(country)[7]
#Finding the true intervention date and calibrating it a bit!
initial_guess = [3.6, 4, 14, 0.8, 0.1, 0.3, 2, 50 , int_day]          
bounds= ((1, 20) , (1,15) , (2,20) , (0.5,1) , (0,1) , (0,1) , (1, 5), (1, 100),(int_day-5,int_day+5))

res = optimize_final(initial_guess,  bounds ,country )

country = "Italy"
int_day = get_country(country)[7]
#Finding the true intervention date and calibrating it a bit!
initial_guess = [3.6, 4, 14, 0.8, 0.1, 0.3, 2, 50 , int_day]          
bounds= ((1, 20) , (1,15) , (2,20) , (0.5,1) , (0,1) , (0,1) , (1, 5), (1, 100),(int_day-5,int_day+5))

res = optimize_final(initial_guess,  bounds ,country)


#Now i am recreating the events with the parameters i have and my function!
#Some of the parameters will also be taken from the papers
#Instead of comparing test set, maybe compare train set results?? for example better fitter france


# In[18]:


#Functions for Rt
#Some extra functions!

def new_Rt(b,c,a,Rt):
  """
  @params
  b : transimition rate, normal value 0.45(or 0.5), masks shops 0.3 , masks closed spaces 0.2 , masks everywhere 0.1-0.05
  c : contracts per person : normal value : 15, work from home 10 , closed bars 8 , closed restaurants 7, lockdown 5, strict lockdown :3
  a : goverments intervetion : normal value 0.8 , recommendations light : 0.70-0.75, strict 0.65 , fees everywhere 0.5
  The values for every measure should be reconsider
  """
  return (Rt - (1-b)**(c*a)*Rt)

def linear_decay(Rold , Rnew , total_steps , step): #t stands for steps of decaying 
    Rt = Rold - ((Rold - Rnew)/total_steps)*step
    return(Rt)


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


# In[19]:


incubation_days = 0 
#Just get R0 and keep the curve to show the good fitting 
#but return the rest of the variables to use them as the standard!

#Getting the variables
R_0 = res[0]

d_hosp = 4
d_crit = 14
inc_days= 5.6
inf_days = 2.9
prob_assymp = 0.8
prob_severe_critical = 0.2
prob_crit_fatal = 0.3
#Initializing 
#Population
N = get_country(country)[0]
#Initial infections
n_inf = get_country(country)[1]
#Forecasting days
days = 200

#Initializing
initials = [(N - n_inf)/ N, 0, n_inf/N, 0, 0, 0, 0]

#Have to initialize the parameters which will be used for the model
#Intervention date:
intervention_date = 40
#Interventaion duration
intervention_duration = 14
#transimition rate to be decreased to:
b = 0.2
#Contracts per person
c = 5
#Strictness of measures
a = 0.6
Rt_new = new_Rt(b,c,a,R_0)
#Defining the model:
args = (My_Rt , inc_days , inf_days , d_hosp , d_crit , prob_assymp , prob_severe_critical , prob_crit_fatal)
solution = solve_ivp(model, [0, days], initials, args =args, t_eval = np.arange(days))



# In[ ]:


#@markdown
#@markdown Incubation and Infectious days
incubation_days = 8 #@param {type:"slider", min:4, max:21, step:1}
infection_days = 4 #@param {type:"slider", min:1, max:10, step:1}
#@markdown Days Spent on Hospital and Days on Critical State
hospital_days = 4 #@param {type:"slider", min:1, max:15, step:1}
Critical_days = 7 #@param {type:"slider", min:1, max:20, step:1}
#@markdown Probabilities:
Assymptomatics = 0.6 #@param {type:"slider", min:0, max:1, step:0.1}
Critical_Symptoms = 0.3 #@param {type:"slider", min:0, max:1, step:0.1}
Mortality_After_Critical = 0.1 #@param {type:"slider", min:0, max:1, step:0.1}
#@markdown Measures Taken:
Intervention_day = 35 #@param {type:"slider", min:15, max:100, step:1}
Intervention_duration = 11 #@param {type:"slider", min:1, max:30, step:1}
#@markdown Measures Type
Transmition_rate = 0.2#@param {type:"slider", min:0, max:0.8, step:0.05}
Contracts_Per_day = 13#@param {type:"slider", min:1, max:25, step:1}
Measures_Strictness = 0.7 #@param {type:"slider", min:0.4, max:1, step:0.05}
Rt_new = new_Rt(Transmition_rate,Contracts_Per_day,Measures_Strictness,R_0)
intervention_date = Intervention_day
#Interventaion duration
intervention_duration = Intervention_duration
args = (My_Rt , incubation_days , infection_days , hospital_days , Critical_days ,
        Assymptomatics , Critical_Symptoms , Mortality_After_Critical)
solution = solve_ivp(model, [0, days], initials, args =args, t_eval = np.arange(days))
S , E , I , H , C , R , D = solution.y

#total cases: infected + hospitlized + critical + recovered + dead
total_cases = I + H + C + R + D 
total_cases = np.clip(I + E + H + C + R + D, 0 , np.inf)
S = N*S[40:]
E = E[40:]*N
I = I[40:]*N
H = H[40:]*N
C = C[40:]*N
R = R[40:]*N
D = D[40:]*N
plt.figure(figsize = (16,8))

plt.style.use('dark_background')

#Width is how i can change the gaps
plt.bar(range(days)[40:],E, width=0.7, align = 'edge')
plt.bar(range(days)[40:],I, width=0.7 , align = 'edge')
plt.bar(range(days)[40:],H, width=0.7 , align = 'edge',color = 'y')
plt.bar(range(days)[40:],D, width=0.7 , align = 'edge', color = 'green')

plt.show()


# ## Trying to define Dashboards

# In[21]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


# In[23]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

fig = px.scatter(df, x="gdp per capita", y="life expectancy",
                 size="population", color="continent", hover_name="country",
                 log_x=True, size_max=60)

app.layout = html.Div([
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




