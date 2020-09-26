import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('covid_india.csv')

data = dataset.iloc[:,1:].values

#dictionary for confirmed cases
dict_data_confirm = {}

#calculate confirm cases per days
for date in data:
	if date[0] in dict_data_confirm.keys():
		dict_data_confirm[date[0]] = dict_data_confirm[date[0]] + date[-1]

	else:
		dict_data_confirm[date[0]] = date[-1]

x_data = []
y_confirm_data = []

for k,v in dict_data_confirm.items():
	x_data.append(k)
	y_confirm_data.append(v)

x_train = []
for i in range(217):
	x_train.append(i)

x_train1 = np.array(x_train).reshape(-1,1)

#convert into polynomial
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x_train1)

y_confirm_train = np.array(y_confirm_data)

#linear regression
model = LinearRegression()

#fit the parameters to the model
poly.fit(x_poly, y_confirm_train)
model.fit(x_poly, y_confirm_train)

#predict the value of corona confirm cases at 220th day
pred = model.predict(poly.fit_transform([[220]]))
print("The confirm cases at 220th day in india is: " + str(int(pred)))

#plot the graph
plt.scatter(x_train, y_confirm_train, color='blue')
plt.xlabel("Number of Days")
plt.ylabel("Number of confirm cases")
plt.plot(x_train, model.predict(x_poly), color='red')
plt.show()

#dictionary for calculating deaths
dict_data_deaths = {}

#calculate death per days
for date in data:
	if date[0] in dict_data_deaths.keys():
		dict_data_deaths[date[0]] = dict_data_deaths[date[0]] + date[-2]

	else:
		dict_data_deaths[date[0]] = date[-2]

y_death_data = []
for i in dict_data_deaths.values():
	y_death_data.append(i)

y_death_train = np.array(y_death_data)

#fit the paramters for death
poly.fit(x_poly, y_death_train)
model.fit(x_poly, y_death_train)

#prediction of no of death on 220th day
pred = model.predict(poly.fit_transform([[220]]))
print("The number of deaths on 220th day in india is: " + str(int(pred)))

#plot the graph of death
plt.scatter(x_train, y_death_data, color='red')
plt.xlabel("Number of Days")
plt.ylabel("Number of deaths")
plt.plot(x_train, model.predict(x_poly))
plt.show()

#now see confirm cases state wise

#first see the connfirm cases and death in kerala
dict_confirm_kerala = {}

#calculate confirm cases in kerala day wise
for date in data:
	if 'Kerala' == date[2]:
		if date[0] in dict_confirm_kerala.keys():
			dict_confirm_kerala[date[0]] = dict_confirm_kerala[date[0]] + date[-1]
		else:
			dict_confirm_kerala[date[0]] = date[-1]

y_confirm_kerala = []

for value in dict_confirm_kerala.values():
	y_confirm_kerala.append(value)

y_kerala_conf_train = np.array(y_confirm_kerala)

#perform polynomial regression
poly.fit(x_poly, y_kerala_conf_train)
model.fit(x_poly, y_kerala_conf_train)

#predict the number of confirmed cases on 220th day
pred = model.predict(poly.fit_transform([[220]]))
print("Confirm cases may reach to " + str(int(pred)) + " in kerala")

#plot the graph of confimred cases in kerala
plt.scatter(x_train, y_kerala_conf_train, color='blue')
plt.plot(x_train, model.predict(x_poly), color='red')
plt.xlabel('Number of Days')
plt.ylabel('Confirm cases in Delhi')
plt.show()

#number of deaths in kerala
dict_deaths_kerala = {}

for date in data:
	if 'Kerala' == date[2]:
		if date[0] in dict_deaths_kerala.keys():
			dict_deaths_kerala[date[0]] = dict_deaths_kerala[date[0]] + date[-2]
		else:
			dict_deaths_kerala[date[0]] = date[-2]

y_death_kerala = []

for value in dict_deaths_kerala.values():
	y_death_kerala.append(value)

y_death_kerala_train = np.array(y_death_kerala)

#apply polynomial regression
poly.fit(x_poly, y_death_kerala_train)
model.fit(x_poly, y_death_kerala_train)

#predict number of deaths on 220th day in kerala
pred = model.predict(poly.fit_transform([[220]]))
print("The number of deaths on 220th day may reach to " + str(int(pred)) + " in kerala")

#plot the graph
plt.scatter(x_train1, y_death_kerala_train, color='blue')
plt.plot(x_train, model.predict(x_poly), color='red')
plt.xlabel('Number of days')
plt.ylabel('Number of deaths in Kerala')
plt.show()

#confirm cases in Delhi
dict_confirm_delhi = {}

#calculate confirm cases in Delhi day wise
for date in data:
	if 'Delhi' == date[2]:
		if date[0] in dict_confirm_delhi.keys():
			dict_confirm_delhi[date[0]] = dict_confirm_delhi[date[0]] + date[-1]
		else:
			dict_confirm_delhi[date[0]] = date[-1]

y_confirm_delhi = []

for value in dict_confirm_delhi.values():
	y_confirm_delhi.append(value)

y_delhi_conf_train = np.array(y_confirm_delhi)

x_confirm_delhi = []

for i in range(185):
	x_confirm_delhi.append(i)

x_delhi_train = np.array(x_confirm_delhi)
x_confirm_delhi = np.array(x_confirm_delhi).reshape(-1,1)

poly_delhi = PolynomialFeatures(degree=3)
x_poly_Delhi_confirm = poly_delhi.fit_transform(x_confirm_delhi)

#perform polynomial regression
poly_delhi.fit(x_poly_Delhi_confirm, y_delhi_conf_train)
model.fit(x_poly_Delhi_confirm, y_delhi_conf_train)

#plot the graph of confimred cases in delhi
plt.scatter(x_delhi_train, y_delhi_conf_train, color='red')
plt.plot(x_delhi_train, model.predict(x_poly_Delhi_confirm))
plt.xlabel('Number of days')
plt.ylabel('Number of confirm cases in delhi')
plt.show()

#predict the number of confirmed cases on 220th day
pred = model.predict(poly_delhi.fit_transform([[220]]))
print("Confirm cases may reach to " + str(int(pred)) + " in delhi")

#deaths in delhi
dict_deaths_delhi = {}

#calculate deaths in delhi day wise
for date in data:
	if 'Delhi' == date[2]:
		if date[0] in dict_deaths_delhi.keys():
			dict_deaths_delhi[date[0]] = dict_deaths_delhi[date[0]] + date[-2]
		else:
			dict_deaths_delhi[date[0]] = date[-2]

y_death_delhi = []

for values in dict_deaths_delhi.values():
	y_death_delhi.append(values)

y_death_delhi = np.array(y_death_delhi)

poly_death_delhi = PolynomialFeatures(degree=3)
x_poly_death_delhi = poly_death_delhi.fit_transform(x_confirm_delhi)

#perform polynomial regression
poly_delhi.fit(x_poly_death_delhi, y_death_delhi)
model.fit(x_poly_death_delhi, y_death_delhi)

#predict possible deaths by 220 days
pred = model.predict(poly_delhi.fit_transform([[220]]))
print("Deaths may reach to " + str(int(pred)) + " in delhi")

#plot the graph of confimred cases in delhi
plt.scatter(x_delhi_train, y_death_delhi, color='red')
plt.plot(x_delhi_train, model.predict(x_poly_death_delhi))
plt.xlabel('Number of days')
plt.ylabel('Number of deaths in delhi')
plt.show()

#confirmed cases in gujarat
dict_confirm_gujarat = {}

for date in data:
	if 'Gujarat' == date[2]:
		if date[0] in dict_confirm_gujarat.keys():
			dict_confirm_gujarat[date[0]] += date[-1]

		else:
			dict_confirm_gujarat[date[0]] = date[-1]

y_confirm_gujarat = []

for value in dict_confirm_gujarat.values():
	y_confirm_gujarat.append(value)

y_confirm_gujarat = np.array(y_confirm_gujarat)

x_train_gujarat = []
for i in range(167):
	x_train_gujarat.append(i)

x_train_gujarat1 = np.array(x_train_gujarat).reshape(-1,1)

poly_gujarat = PolynomialFeatures(degree=3)
poly_gujarat_train = poly_gujarat.fit_transform(x_train_gujarat1)

#perform polynomial regression
poly_gujarat.fit(poly_gujarat_train, y_confirm_gujarat)
model.fit(poly_gujarat_train, y_confirm_gujarat)

#predict confirm cases in gujarat by 220 days
pred = model.predict(poly_gujarat.fit_transform([[220]]))
print("Confirm cases may reach to " + str(int(pred)) + " in gujarat")

#plot confirm cases of gujarat in graph
plt.scatter(x_train_gujarat, y_confirm_gujarat)
plt.plot(x_train_gujarat, model.predict(poly_gujarat_train), color='red')
plt.xlabel('Number of days')
plt.ylabel('Number of confirm cases in gujarat')
plt.show()

#deaths in gujarat
dict_deaths_gujarat = {}

for date in data:
	if 'Gujarat' == date[2]:
		if date[0] in dict_deaths_gujarat.keys():
			dict_deaths_gujarat[date[0]] += date[-2]

		else:
			dict_deaths_gujarat[date[0]] = date[-2]

y_death_gujarat =  []
for value in dict_deaths_gujarat.values():
	y_death_gujarat.append(value)

y_death_gujarat = np.array(y_death_gujarat)

#by viewing the graph of deaths in gujarat it increase linearly
#so here we can not perform polynomial regression
#we use linear regression as our mode
model1 = LinearRegression()
model1.fit(x_train_gujarat1, y_death_gujarat )

#predict death in gujarat by 220 days
pred = model1.predict([[200]])
print("Number of deaths may reach to " + str(int(pred)) + " in gujarat")

#plot confirm cases of gujarat in graph
plt.scatter(x_train_gujarat, y_death_gujarat)
plt.plot(x_train_gujarat, model1.predict(x_train_gujarat1), color='red')
plt.xlabel('Number of days')
plt.ylabel('Number of confirm cases in gujarat')
plt.show()
