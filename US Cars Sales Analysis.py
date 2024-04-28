#!/usr/bin/env python
# coding: utf-8

# # Python Libraries

# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import warnings
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings("ignore")


# # Information about Data

# In[62]:


filepath="C:/Users/kashy/OneDrive/Desktop/Datasets for Python visualizations/Cars Sales Dataset/car_prices.csv"
df=pd.read_csv(filepath)


# In[63]:


df.head(5)


# In[64]:


df.shape


# In[65]:


df.info()


# In[66]:


df.describe().round().T


# In[67]:


missing_values=df.isnull().sum()
missing_values


# # Cleaning Data
# 

# In[68]:


df.dropna(inplace=True)


# In[69]:


df.drop(columns="saledate")


# In[70]:


df.drop(columns="vin")


# In[71]:


df.isnull().sum()


# # Data Visualization

# In[72]:


top_10_models=df.groupby("model").agg({"sellingprice":"mean"}).sort_values("sellingprice", ascending = False).iloc[:5]


top_10_models=top_10_models.head(10)
top_10_models


# In[73]:


plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
cols = ['orange' if y < 175000 else 'green' for y in top_10_models.sellingprice]
ax=sns.barplot(x=top_10_models.index,y=top_10_models['sellingprice'],data=top_10_models,palette=cols)
ax.bar_label(ax.containers[0])



# 458 Italia is the top car model by aggregated mean of selling price of 183000

# # TOP 5 CARS COMPANIES BY SELLING PRICE

# In[74]:


top_10_make=df.groupby("make").agg({"sellingprice":"mean"}).sort_values("sellingprice", ascending = False).iloc[:5]


top_10_make=top_10_make.head(10)
top_10_make.round()


# In[75]:


plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
plt.pie(top_10_make['sellingprice'], labels=top_10_make.index, autopct='%1.1f%%', startangle=90)
plt.title('Selling Price Distribution by Car Company')
plt.axis('equal')  
plt.show()


# Rolls Royce has the Highest distribution in selling price of around 28.8%

# In[76]:


top_10_color=df.groupby("color").agg({"sellingprice":"mean"}).sort_values("sellingprice",ascending=False).iloc[:10]


top_10_color=top_10_color.head(10)
top_10_color.round()


# In[77]:


df['color'].unique()


# In[78]:


filtered_df = top_10_color[top_10_color.index != "—"]
sns.set_style("whitegrid")
plt.figure(figsize=(9, 8))
palette="Set2"
ax=sns.barplot(
    filtered_df,
    x=top_10_color.index,
    y=top_10_color["sellingprice"],
    palette=palette,
)

plt.xlabel("Color Category")
plt.ylabel("Selling Price")
plt.title("Selling Price by Color")
ax.bar_label(ax.containers[0])


# The most popular car colors on the market today are neutral tones like charcoal. If you're aiming for the quickest sale or the best resale value, consider choosing a neutral color.

# In[79]:


car_model_values = df["make"].value_counts().sort_values(ascending=False).iloc[0:10]
val= car_model_values.values
model = car_model_values.index


# In[85]:


plt.figure(figsize=(9, 8))
sns.set_style("whitegrid")
palette="Set2"
ax=sns.barplot(
    x=model,
    y=val,
    palette="Accent")
plt.xlabel("Top 10 Models")
plt.ylabel("Values")
plt.title("Distribution of Value for Top 10 Car Models")
ax.bar_label(ax.containers[0])


# Ford is a top-selling car brand, particularly known for its popular F-Series trucks, which have been the best-selling vehicle in the United States for over four decades.

# In[86]:


year_df = df.groupby(by="year", as_index=False)["sellingprice"].first()


# In[87]:


year_df


# In[88]:


fig = px.line(year_df, x="year", y="sellingprice")
fig.show()


# Car sales have been surging, and 2012 saw a champion emerge from the pack! Let's discover which car took the top spot in that booming year.¶

# In[89]:


data_by_seller = df.groupby(by=["seller", "year"], as_index=False)[
    "sellingprice"
].first()
data_by_seller.sort_values(by="sellingprice", ascending=False, inplace=True)


# In[90]:


max_price_index = data_by_seller["sellingprice"].idxmax()
row_with_max_price = data_by_seller.loc[max_price_index]

seller_with_max_price = row_with_max_price["seller"]
max_selling_price = row_with_max_price["sellingprice"]
year_of_max_price = row_with_max_price["year"]

print(
    f"The seller, {seller_with_max_price}, bought at the maximum selling price of {max_selling_price:.1f} in year {year_of_max_price}."
)


# In[91]:


make_price = df.groupby(by="make", as_index=False)["sellingprice"].first()
make_price.sort_values(by="sellingprice", ascending=False, inplace=True)


# In[92]:


make_price


# In[93]:


max_price_make = make_price.head(1)["make"].values[0]

min_price_make = make_price.tail(1)["make"].values[0]

print("Car make with the maximum selling price:", max_price_make)
print("Car make with the minimum selling price:", min_price_make)


# In[94]:


df.columns


# In[95]:


new_df=df.groupby(by=['year','make','transmission','color','odometer','mmr'],as_index=False,)["sellingprice"].first()
new_df.sort_values(by="sellingprice",ascending=False).head(5)


# In[96]:


new_df.shape


# In[97]:


yearly_mean_price=(new_df.groupby(by=['year'],as_index=False,)['sellingprice'].mean().round(2))


# In[98]:


yearly_mean_price


# In[99]:


fig=px.bar(yearly_mean_price,x='year',y='sellingprice')
fig.update_layout(title_text="Mean Selling Price by Year")
fig.show()


# In[100]:


top_makes_per_year=new_df.groupby("year",as_index=False).apply(lambda x:x.nlargest(1,"sellingprice",keep='all'))


# In[101]:


top_makes_per_year


# In[102]:


top_makes_per_year=top_makes_per_year.rename(columns={"sellingprice":"HighestMeanPrice"})


# In[103]:


sns.barplot(top_makes_per_year,x="year",y="HighestMeanPrice",palette="Set2")
plt.xticks(rotation=90)
plt.title("Mean Price of top makes per Year")
plt.show()


# In[104]:


values=df['transmission'].value_counts()


# In[105]:


transmission_types=df['transmission'].unique()


# In[106]:


transmission_types


# In[107]:


fig = go.Figure(data=[go.Pie(labels=transmission_types, values=values, pull=[0, 0.2],title="Transmission Distribution")])
fig.show()


# The most popular cars by transmission are of Automatic 

# In[108]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(df, x="mmr", y="sellingprice", color="#08677a", ax=ax[0])
sns.scatterplot(df, x="odometer", y="sellingprice", color="#bc1618", ax=ax[1])

plt.xlabel("MMR")
plt.ylabel("Selling Price")
plt.xlabel("Odometer (km)")
plt.ylabel("Selling Price")
plt.suptitle("Selling Price vs. MMR and Odometer")
plt.tight_layout()

plt.show()


# From above we can conclude
# Two key factors that can increase a car's selling price are lower mileage and high market demand for that specific model.
# For the best selling price, aim for a car with both low mileage and strong market demand.
# 

# In[109]:


sns.relplot(df,x="condition",y="sellingprice",kind="line",linewidth=2,marker="o",markersize=8,alpha=0.7,dashes=False,legend="full",markerfacecolor="#262182",)
plt.xlabel("Condition")
plt.ylabel("Selling Price")
plt.title("Selling Price vs. Condition")


plt.show()


# The most popular cars for resale are often those that are relatively new (around 5 years old) and have a strong reputation for reliability (rated close to 50 on a 100-point scale)

# In[110]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sns.set_style('whitegrid')


# In[111]:


sns.boxplot(x='transmission', y='condition', data=df, palette='hls')


# # Find Correlation between Variables

# In[112]:


import scipy
from scipy.stats.stats import pearsonr


# In[113]:


df.columns


# In[114]:


X = df[['condition', 'odometer', 'mmr', 'sellingprice']]
sns.pairplot(X)


# In[115]:


correlation_matrix = X.corr()


# In[116]:


correlation_matrix


# In[117]:


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot')
plt.show()


# # Training and Testing the Model

# Using the correlation matrix we have determine the there is a strong correlation between mmr and selling price of 0.98

# In[118]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming you have your data in arrays X (features) and y (target)
# Example data
X = df[['condition', 'odometer', 'mmr']]

y =df["sellingprice"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[119]:


from sklearn.linear_model import LinearRegression
clf=LinearRegression()



# In[120]:


clf.fit(X_train,y_train)


# In[121]:


clf.predict(X_test)


# In[122]:


clf.predict(X_train)


# In[123]:


y_test


# In[124]:


clf.score(X_test,y_test)


# In[125]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(df['mmr'],df['sellingprice'],color='red',marker='+')
plt.xlabel('Highway-mpg')
plt.ylabel('Price')


# In[126]:


X=df[['mmr']]
Y=df['sellingprice']


# In[127]:


lm=LinearRegression()
lm.fit(X,Y)


# In[128]:


plt.xlabel('mmr')
plt.ylabel('sellingprice')
plt.scatter(df['mmr'],df['sellingprice'],color='red',marker='+')
plt.plot(df['mmr'],lm.predict(X),color='blue')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




