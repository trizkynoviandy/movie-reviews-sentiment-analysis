import joblib
import streamlit as st

loaded_model = joblib.load("model/movie_review_model.sav")

st.title("Movie Review Sentiment Prediction")
st.markdown("""
            ### Predict the sentiment in a movie review
            * This app used logistic regression to train this model, with 91% and 89% accuracy on the training set and the testing set, respectively
            
            * Dataset source: <http://ai.stanford.edu/~amaas/data/sentiment/>
            """)

text_input = st.text_area(label="Input a movie review", height=250)
predict = st.button("Predict")

if predict:
    try:
        prediction = loaded_model.predict(list([text_input]))
        st.success('Prediction Succesful!')
        if str(prediction) == "[0]":
            st.write("Result: This is a negative review")
        else:
            st.write("Result: This is a positive review")
    except ValueError:
        st.write("Make sure your data is correct")