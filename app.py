import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.title("Food Recommendation System")
st.text("Let us help you with ordering")
st.image("foood.jpg")

## nav = st.sidebar.radio("Navigation",["Home","IF Necessary 1","If Necessary 2"])

st.subheader("Whats your preference?")
vegn = st.radio("Vegetables or none!", ["veg", "non-veg"], index=1)

st.subheader("What Cuisine do you prefer?")
cuisine = st.selectbox("Choose your favourite!", ['Healthy Food', 'Snack', 'Dessert', 'Indian', 'Chinese', 'Beverage'])


st.subheader("How well do you want the dish to be?")  #RATING
val = st.slider("from poor to the best!",0,5)

#food = pd.read_csv("input/food.csv")
#ratings = pd.read_csv("input/ratings.csv")
#combined = pd.merge(ratings, food, on='Food_ID')

##ans = food.loc[(food.C_Type == cuisine) & (food.Veg_Non == vegn),['Name','C_Type','Veg_Non']]
combined = pd.read_csv("reviews.csv")


ans = combined.loc[(combined.rating >= val),['name']]
names = ans['name'].tolist()
x = np.array(names)
ans1 = np.unique(x)

finallist = ""
bruh = st.checkbox("Choose your Dish")
if bruh == True:
    finallist = st.selectbox("Our Choices", ans1)


##### IMPLEMENTING RECOMMENDER ######
#dataset = ratings.pivot_table(index='Food_ID',columns='User_ID',values='Rating')
dataset = combined.pivot_table(index='itemid', columns='userid', values='rating')
dataset.fillna(0,inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)


def food_recommendation(Food_Name):
    n = 5
    FoodList = combined[combined['name'].str.contains(Food_Name)]
    if len(FoodList):
        Foodi= FoodList.iloc[0]['itemid']
        Foodi = dataset[dataset['itemid'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)
        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['itemid']
            i = combined[combined['itemid'] == Foodi].index
            Recommendations.append({'name':combined.iloc[i]['name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['name']
    else:
        return "No Similar Foods."

display = food_recommendation(finallist)
#names1 = display['Name'].tolist()


#recoomended item img pickup
#print(display)
res = pd.read_csv('products.csv')
res = res.drop(columns=['_id', 'countInStock', 'createdAt', 'description', 'isActive', 'numReviews', 'price', 'public_id', 'updatedAt', 'reviews'])
pd.set_option("display.max_colwidth", 100)
res.dropna(inplace=True)
#print(res)


#x1 = np.array(names)
#ans2 = np.unique(x1)
#old
if bruh == True:
    bruh1 = st.checkbox("We also Recommend : ")
    if bruh1 == True:
        for i in display:
            x = res[res['name'].isin([i])]
            st.image(x['image'].to_string(index=False), width=128)
            st.write(i)

