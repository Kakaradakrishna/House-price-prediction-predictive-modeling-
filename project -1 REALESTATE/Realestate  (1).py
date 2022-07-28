import streamlit as st

import pandas as pd

import joblib

data = pd.read_csv(r"C:\Users\kakar\Downloads\DS Data sets\Realestate_with_clusters.csv")

classification_model = joblib.load("RF_class.pkl")
regression_model= joblib.load("RF_regression.pkl")
ohe = joblib.load("ohe_class.joblib")



def main():
	st.title("Price Estimation of Properties")
	st.subheader("Data Snippet")
	st.dataframe(pd.read_csv('cleaned_nafill.csv'))
	cola, colb = st.columns(2)
	with cola:
		if st.button("EDA (Exploratory Data Analysis"):
			st.write(data.describe())

	
	st.subheader("Fill up the values to estimate Property")

	col1, col2 = st.columns(2)
	d = {}
	for i in data.state.unique():
		d[i] = list(set(data.city[data.state == i]))
	with col1:
		Hyderabaad_city = st.selectbox('Locality',d.keys())

	col2, col3 = st.columns(2)
	with col2:
		Age_of_Property = st.selectbox("Age of Property (Years)", data.house_type.unique())
	with col3:
		sqft = st.number_input("Area")

	col4, col5 = st.columns(2)
	with col4:
		NO_OF_Bedrooms = st.number_input("Number of Rooms")
	with col5:
		Bathrooms = st.number_input("Enter no of Bathrooms")

	col6, col7 = st.columns(2)
	with col6:
		furnishing_status = st.selectbox("Select Furinishing Type", data.furnishing.unique())
	with col7:
		constr_info = st.selectbox("Possession Status",data.constr_info.unique())

	col8,col9 = st.columns(2)
	with col8:
		floor_no = st.number_input("Total Floor Count")Enter Floor no
    with col9:
        Select_Facing =st.selectbox("View /Facing",data.constr_info.unique())
        
	if st.button("Estimate Price"):
        ## regression
		values = [[Hyderabaad_city,Age_of_Property,sqft,NO_OF_Bedrooms,Bathrooms,furnishing_status,constr_info,floor_no,Select_Facing]]
		check = pd.DataFrame(values, columns = ['Locality', 'Area', 'Possession Status', 'Furnishing Status', 'Number of Rooms', 'Number of Bathroom',
       'Number of Parking', 'View /Facing','Age of Property (Years)','Total Floor Count'])
		cats = pd.DataFrame(ohe.transform(check.iloc[:,[0,2,3,7,9,10]]))
		cats.columns = ohe.get_feature_names_out()
		record = pd.concat([check.iloc[:,[1,4,5,6,8,11]],cats], axis=1)
		st.write("Estimated Price Value",":",round(Regression_model.predict(record)[0],0))
       
    ## Classfication
        
		values2 = [[Hyderabaad_city,Age_of_Property,sqft,NO_OF_Bedrooms,Bathrooms,furnishing_status,constr_info,floor_no,Select_Facing]]
		check2 = pd.DataFrame(values2, columns = ['Locality', 'Area', 'Possession Status', 'Furnishing Status', 'Number of Rooms', 'Number of Bathroom',
       'Number of Parking', 'View /Facing','Age of Property (Years)','Total Floor Count'])
		cats2 = pd.DataFrame(ohe.transform(check2.iloc[:,[0,2,3,7,9,10]]))
		cats2.columns = ohe.get_feature_names_out()
		record = pd.concat([check.iloc[:,[1,4,5,6,8,11]],cats], axis=1)
		st.write("Estimated Price Value",":",round(classfication_model.predict(record)[0],0))

if __name__=="__main__":