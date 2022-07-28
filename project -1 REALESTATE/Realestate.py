import streamlit as st

import pandas as pd

import joblib

import base64
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack('rrr.jpg')

data = pd.read_csv("Realestate_with_clusters.csv")

classification_model = joblib.load("RF_class.pkl")
regression_model= joblib.load("RF_regression.pkl")
ohe = joblib.load("ohe_class.joblib")



def main():
	st.title("Price Estimation of Properties")
	st.subheader("Data Snippet")
	st.dataframe(pd.read_csv('Realestate_with_clusters.csv'))
	
	st.subheader("Fill up the values to estimate Property")


	Locality = st.selectbox('Locality',data.Locality.unique())

	col2, col3 = st.columns(2)
	with col2:
		Age_of_Property = st.selectbox("Age of Property (Years)",data['Age of Property (Years)'].unique())
	with col3:
		area = st.number_input("Area")

	col4, col5 = st.columns(2)
	with col4:
		NO_OF_Bedrooms = st.number_input("Number of Rooms")
	with col5:
		Bathrooms = st.number_input("Enter no of Bathrooms")

	col6, col7 = st.columns(2)
	with col6:
		furnishing_status = st.selectbox("Select Furinishing Type",data['Furnishing Status'].unique())
	with col7:
		posst = st.selectbox("Possession Status",data['Possession Status'].unique())

	col8,col9 = st.columns(2)
	with col8:
		floor_no = st.number_input("Total Floor Count")
	with col9:
		Select_Facing =st.selectbox("View /Facing",data['View /Facing'].unique())

	col10,col11=st.columns(2)
	with col10:
	     no_parkings=st.number_input("Enter Parking Slots")
	with col11:
	     additional_rooms=st.selectbox("Additional Rooms",data['Additional Rooms'].unique())


	Floor_No =st.number_input("Enter Floor Number")
    	
        
	if st.button("Estimate Price"):
		## regression
		values = [[Locality,Age_of_Property,area,NO_OF_Bedrooms,Bathrooms,furnishing_status,posst,floor_no,
		Select_Facing,no_parkings,additional_rooms,Floor_No]]
		check = pd.DataFrame(values, columns = ['Locality', 'Age of Property (Years)', 
			'Area','Number of Rooms','Number of Bathroom','Furnishing Status',
			'Possession Status','Total Floor Count','View /Facing','Number of Parking','Additional Rooms','Floor No.'])
		cats = pd.DataFrame(ohe.transform(check.iloc[:,[0,1,5,6,8,10]]))
		cats.columns = ohe.get_feature_names()
		record = pd.concat([check.iloc[:,[2,3,4,7,9,11]],cats], axis=1)
		st.write("Estimated Price Value",":",round(regression_model.predict(record)[0],0))

		class_=[]
		## Classfication
		if classification_model.predict(record)[0]==0:
			class_.append('low price')
		elif classification_model.predict(record)[0]==1:
			class_.append('High')
		else:
			class_.append('premium')	

		st.write("Estimated Price Class",":",class_[0])

if __name__=="__main__":
	main()