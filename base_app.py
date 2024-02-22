"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# ... (previous imports and variable assignments)

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("FM1 Tweet Classifier")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from the supporting resources folder
        st.markdown(
            "This process requires the user to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe in climate change. Below you will find information about the data source and a brief data description. You can have a look at word clouds and other general EDA on the EDA page, and make your predictions on the prediction page that you can navigate to in the sidebar."
        )

        st.subheader("Raw Twitter data and label")
        # Use st.file_uploader to allow users to upload a CSV file
        uploaded_file = st.file_uploader("resources/train.csv", type=["csv"])

        # Check if a file is uploaded
        if uploaded_file is not None:
            # Read the CSV file into a DataFrame
            uploaded_data = pd.read_csv(uploaded_file)

            # Display the raw data if checkbox is checked
            if st.checkbox("Show raw data"):
                st.write(uploaded_data[["sentiment", "message"]])
        else:
            st.warning("Please upload a CSV file.")

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("resources/Logistic_regression.pkl"), "rb")
            )
            prediction = predictor.predict(vect_text)

            # When the model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))


# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
