# THis program analyses house prices for a certain region based on public data availalbe and can predict the approximate 
# price of your house. Make sure the home_data.gl file is unzipped and placed in the same folder as the Predict_House_Price.py file.

#Fire up modules
import graphlab
import matplotlib.pyplot as plt
%matplotlib inline

#Load some house sales data
sales = graphlab.SFrame('home_data.gl/')

#Display
sales

#Exploring the data for housing sales
#graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")

#Create a simple regression model of sqft_living to price
train_data,test_data = sales.random_split(.8,seed=0)

##Build the regression model using only sqft_living as a feature
sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)

#Evaluate the simple model
print test_data['price'].mean()

print sqft_model.evaluate(test_data)

#Display Predictions
plt.plot(test_data['sqft_living'],test_data['price'],'.', test_data['sqft_living'],sqft_model.predict(test_data),'-')
sqft_model.get('coefficients')

#Explore other features in data
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

#Build model with more features
my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)

#Compare results of both models
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

house1 = sales[sales['id']=='5309101200']
print house1['price']
print sqft_model.predict(house1)
print my_features_model.predict(house1)

house2 = sales[sales['id']=='1925069082']
print sqft_model.predict(house2)
print my_features_model.predict(house2)

# Question 1
new_set = sales[sales['zipcode']=='98039']
new_set_mean = new_set['price'].mean()
print new_set_mean

# Question 2
new_set_two = sales[(sales['sqft_living'] >2000) & (sales['sqft_living'] < 4000)]
new_set_two = new_set_two['sqft_living']
frac =  len(new_set_two)*100/ len(sales)
print frac

# Question 3
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)
print  my_features_model.evaluate(test_data)
print advanced_features_model.evaluate(test_data)

