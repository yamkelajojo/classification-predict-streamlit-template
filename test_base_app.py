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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import joblib, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import string
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import warnings
import pickle
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


# Data dependencies
import pandas as pd
nltk.download('stopwords')
nltk.download('omw-1.4')


# Vectorizer
news_vectorizer = joblib.load("betterVect_sentiment.pkl", "rb")
# tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")

st.set_page_config(
    page_title="FeedFrenzy",
    page_icon=":earth_americas:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# tweet_text = st.text_area("Enter Text","Type Here")
# The main function where we will build the actual app
def main(raw=raw):
    """Tweet Classifier App with Streamlit"""
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("FeedFrenzy")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "About", "Information", "EDA"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown(" Data description as per source")
        st.markdown(
            "Data description as per source The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).Each tweet is labelled as one of the following classes:- 2(News): the tweet links to factual news about climate change- 1(Pro): the tweet supports the belief of man-made climate change- 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change- -1(Anti): the tweet does not believe in man-made climate change"
        )

        st.subheader("Raw Twitter data and label")
        if st.checkbox("Show raw data"):  # data is hidden if box is unchecked
            st.write(raw[["sentiment", "message"]])  # will write the df to the page

    if selection == "Prediction":
        st.info("Prediction with ML Models")
        tweet_text = st.text_area(
            "Enter text to see whether it's Neutral, Pro, News or Anti towards climate change",
            "Type Here",
        )

        option_desc = [
            "Neutral: the text neither supports nor refutes the belief of man-made climate change",
            "Pro: the text supports the belief of man-made climate change",
            "News: the text links to factual news about climate change",
            "Anti: the text does not believe in man-made climate change Variable definitions",
        ]

        if st.button("Classify"):
            text = list([tweet_text])
            raw = pd.DataFrame(text, columns=["message"])
            raw["message"].replace("\d+", "", regex=True, inplace=True)

            def remove_RT(column_name):
                return re.sub(r"^rt[^\s]+", "", column_name)

            raw["message"] = raw["message"].apply(remove_RT)
            raw["message"] = raw["message"].str.replace("rt", "")

            def remove_handels(post):
                return re.sub("@[^\s]+", " ", post)

            raw["message"] = raw["message"].apply(remove_handels)
            # removing the url
            pattern_url = r"http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+"
            raw["message"] = raw["message"].replace(
                to_replace=pattern_url, value=" ", regex=True
            )

            def remove_hashtages(post):
                return re.sub("#[^\s]+", " ", post)

            raw["message"] = raw["message"].apply(remove_hashtages)

            def remove_punctuation(post):
                return "".join([l for l in post if l not in string.punctuation])

            raw["message"] = raw["message"].apply(remove_punctuation)
            raw["message"] = raw["message"].str.split()
            stemmer = SnowballStemmer("english")
            raw["message"] = raw["message"].apply(
                lambda x: [stemmer.stem(y) for y in x]
            )

            def remove_stop_words(tokens):
                return [t for t in tokens if t not in stopwords.words("english")]

            raw["message"] = raw["message"].apply(remove_stop_words)
            nltk.download("wordnet")
            lemmatizer = WordNetLemmatizer()

            def mbti_lemma(words, lemmatizer):
                return [lemmatizer.lemmatize(word) for word in words]

            raw["message"] = raw["message"].apply(mbti_lemma, args=(lemmatizer,))
            raw["message"] = raw["message"].apply(" ".join)

            # tweet_text = raw["message"][0]

            # if st.button("submit"):
            # tweet_text = st.text_area("happy")
            tweet_text = tweet_text.lower()
            vect_text = news_vectorizer.transform(np.array([tweet_text]))
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("Climant_change_sentiment.pkl"), "rb")
            )

            prediction = predictor.predict(vect_text)[0]

            if prediction == -1:
                st.success("the text does not believe in man-made climate change")
            elif prediction == 0:
                st.success("the text neither supports nor refutes the belief of man-made climate change")
            elif prediction == 1:
                st.success("the text supports the belief of man-made climate change")
            elif prediction == 2:
                st.success("the text links to factual news about climate change")

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.

    if selection == "EDA":
        sentiment_labels = {
            "-1": "-1:Anti",
            "0": "0:Neutral",
            "1": "1:Pro",
            "2": "2:News",
        }

        # Convert the 'sentiment' column to string type
        raw["sentiment"] = raw["sentiment"].astype(str)

        # Get the sentiment counts
        sentiment_counts = raw["sentiment"].value_counts()

        # Replace the index labels with your sentiment_labels
        sentiment_counts.index = [
            sentiment_labels.get(i, "Unknown") for i in sentiment_counts.index
        ]

        st.bar_chart(sentiment_counts)

        # Extracting hashtags from tweets
        hashtag_list = []
        for message in raw["message"]:
            if message:
                tags = message.split()
                for tag in tags:
                    tag = "#" + tag.strip(",")
                    tag = tag.lower()
                    hashtag_list.append(tag)

        # Plotting hashtag bar graph
        hashtag_counts = Counter(hashtag_list)
        top_hashtags = hashtag_counts.most_common(7)
        hashtags, counts = zip(*top_hashtags)

        fig_hashtags, ax_hashtags = plt.subplots(figsize=(10, 6))
        ax_hashtags.bar(hashtags, counts, color="red")
        ax_hashtags.set_xlabel("Hashtags")
        ax_hashtags.set_ylabel("Count")
        ax_hashtags.set_title("Top 7 Unique Hashtags")
        ax_hashtags.set_xticklabels(hashtags, rotation=45)

        st.pyplot(fig_hashtags)

        # happy
        tweet = raw["message"].iloc[0]
        words = tweet.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        word_counts_df = pd.DataFrame.from_dict(
            word_counts, orient="index", columns=["count"]
        )
        word_counts_df = word_counts_df.sort_values(by="count", ascending=False)

        fig_words, ax_words = plt.subplots(figsize=(10, 6))
        word_counts_df["count"].plot(kind="bar", color="skyblue")
        ax_words.set_title("Word Counts in the Tweet")
        ax_words.set_xlabel("Words")
        ax_words.set_ylabel("Count")
        ax_words.set_xticklabels(word_counts_df.index, rotation=50, ha="right")

        st.pyplot(fig_words)

    if selection == "About":
        st.subheader("About")

        cc_image = "causeseffects-nologo.jpg"

        st.image(
            cc_image,
            caption=None,
            width=None,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )

        st.markdown(
            "Companies focusing on environmental sustainability aim to understand public perceptions of climate change as a potential threat."
        )
        st.markdown(
            " This insight will enhance their market research, helping them anticipate consumer attitudes toward their eco-friendly products and services"
        )

        company_logo = "comp_logo.png"

        employees = [
            {
                "name": "Yamkela",
                "occupation": "Software Engineer",
                "image_path": "yamkela.jpg",
            },
            {"name": "Happy", "occupation": "Data Scientist", "image_path": "anza.jpg"},
            {
                "name": "Thabatha",
                "occupation": "Data Scientist",
                "image_path": "thabatha.jpg",
            },
            {
                "name": "Nompumezo",
                "occupation": "Data Scientist",
                "image_path": "nompumezo.jpg",
            },
            {
                "name": "Noluthando",
                "occupation": "Data Scientist",
                "image_path": "noluthando.jpg",
            },
            {
                "name": "Londeka",
                "occupation": "Data Analyst",
                "image_path": "londeka.png",
            },
            # Add more employee data as needed
        ]
        # Title of the app
        st.title("Employee Cards App")
        # Display employee cards
        for employee in employees:
            # Image
            image_path = employee["image_path"]
            st.image(image_path, caption="", width=150)

            # Name
            st.subheader(employee["name"])

            # Occupation
            st.text(employee["occupation"])

            # Add a separator between employee cards
            st.markdown("---")


if __name__ == "__main__":
    main()
    from PIL import Image


def change_background_color(image_path, new_background_color=(255, 255, 255)):
    """
    Change the background color of an image.

    Parameters:
        image_path (str): Path to the input image.
        new_background_color (tuple): RGB tuple representing the new background color.

    Returns:
        PIL.Image.Image: Image object with the new background color.
    """
    with Image.open(image_path) as img:
        # Convert the image to RGBA mode (if not already)
        img = img.convert("RGBA")

        # Create a new image with the desired background color
        background = Image.new("RGBA", img.size, new_background_color)

        # Composite the original image onto the new background
        composed_img = Image.alpha_composite(background, img)

        # Convert back to RGB mode
        composed_img = composed_img.convert("RGB")

    return composed_img


# # Example usage
# image_path = "example.jpg"  # Path to the input image
# new_background_color = (255, 0, 0)  # New background color (red in this example)
# new_image = change_background_color(image_path, new_background_color)

# # Save or display the new image
# new_image.save("new_image.jpg")  # Save the new image
# new_image.show()  # Display the new image
