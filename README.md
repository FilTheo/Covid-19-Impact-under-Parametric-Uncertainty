# Covid-19-Impact-under-Parametric-Uncertainty
## A forecasting framework considering the uncertainties found in Covid-19 literature regarding its core mechanism

#### This project is the merged work for the courses "Data-Driven-Decision-Making" and "Complex-Data" of the Data Science MSc program at the University of Skövde.
##### The project was completed in October 2020. The framework is designed using plotly-dash in June 2021


It results in a Data-Driven-Decision-Making (DDDM) framework responsible for simulating the spread of Covid-19 in a picked country.
The framework uses a SEIR-HCD epidemiological model to forecast (among others) the number of cases and hospitalizations.

However, as the differential equations composing the epidemiological models include parameters related to core mechanisms of Covid-19, there is a degree of 
parametric uncertainty associated with such forecasts.

By considering the conflicting results on research papers about some fundamental mechanisms (eg Reproduction number, incubation period, infectious period, average hospitalization period, etc) the aforementioned parametric uncertainty is vastly increased. Policymakers responsible for handling the forecasting tools struggle to find the best available 
values to initiate the forecasting epidemiological models. 

The second core component of the framework is a knowledge extraction NLP model. For a specific user query (say incubation period) it investigates the Covid-19 literature corpus 
composed of thousands of scientific papers and returns the most representative range of values (say 6,4,10,12, for incubation period)

For every parameter of the SEIR-HCD model, the whole range of values existing in the literature corpus is extracted using the knowledge extraction NLP model.
Details about the parameters and the complete differential equations are given in the attached report.
These ranges of values are added to the DDDM framework in the form of sliders as shown on the left side of the figure below:

![Alt text](https://github.com/FilTheo/Covid-19-Impact-under-Parametric-Uncertainty/blob/main/framework_example.png?raw=true)

#### In some future work the layout of the framework will be updated and the parametric uncertainty will be visualized (Details about the uncertainty visualization are given in the report)


Users are recommended to initiate the framework by selecting a Country and the Forecasting Horizon.
Then they should pick the scenario they would like to investigate by picking the initial reproduction number R0. 
The reproduction number R0 reflects how much has the virus been spread in the community. Its initial value reflects the scenario users want to approach and is related to the current situation in a country.
A R0 over 2.5 indicates that the virus is well spread inside the community. In this scenario, users should identify a combination of measures, such as a lockdown or the obligatory use of masks, so the spread is reduced as soon as possible. 
A R0 between 1 and 1.5 indicates that the situation is under control. Users should ensure the situation remains stable without applying intense measures. Before exploring the different mechanisms of the framework, users are recommended to pick an initial reproduction number $R_0$.

Once the country, the forecasting horizon, and the initial reproduction number R0 are selected, a forecast is produced.
![Alt text1](https://github.com/FilTheo/Covid-19-Impact-under-Parametric-Uncertainty/blob/main/framework_example1.png?raw=true)

By manipulating the parameters under the Pandemic Dynamics options the nature of the forecasts is inherently different as the differential equations are altered.
Users will be able to examine the effect when the dynamics of the virus are changing.

**Interventions** 

Based on the aforementioned, it becomes obvious that for identical scenarios different outcomes are possible (eg the total number of new cases is naturally different with a different infectious period per person)
With that said, policymakers should plan different strategies to deal with different outcomes.

To test the effect different measures would have on halting the spread of the virus under specific parametric values, the framework includes an "Intervention Option List"
Users will be able to apply interventions translated into decays in the reproduction number.

The reduction rate of the reproduction number Rt is given by an invented formula:

<img src="https://render.githubusercontent.com/render/math?math=R_tRR = (1-b)^{c*a}"> 

where b is the transmission rate, c the average contacts per day, and a the government's strictness. 
The default values for the above parameters are 0.9 for a, 13 and 0.9 for c and b respectively. A lower value for a points out to a more intense application of the selected measures. 
It should be mentioned that governments' strictness can not drop lower than 0.5.

The new $R_0$ number is given by :

<img src="https://render.githubusercontent.com/render/math?math=NewR_0 = R_0 - R_tRR * R_0"> 

Users can pick different values for a, b, and c that reflects different combinations of measures. In addition, they can pick some predefined recommended combinations as shown below
Different measure combinations are useful in different scenarios.

![Alt text2](https://github.com/FilTheo/Covid-19-Impact-under-Parametric-Uncertainty/blob/main/framework_example2.png?raw=true)

To summarize the description of the project, the aim of the framework is to guide users into investigating the uncertainty associated with Covid-19.
Then they can try different measures to test different simulation scenarios. 

The framework is not completed yet as some bugs are present. 
This is just the rough draft.

In addition, the final layout is also being modified


