# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file
# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Load your trained model
model = joblib.load("resources/train3.pkl")


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox("Show raw data"):  # data is hidden if box is unchecked
            st.write(raw[["sentiment", "message"]])  # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Use the loaded model to make predictions
            prediction = model.predict(vect_text)

            # When model has successfully run, will print prediction
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
