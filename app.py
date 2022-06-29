import streamlit as st

#connect to Google Could Storage
from google.oauth2 import service_account
from google.cloud import storage

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
from tensorflow import keras
import tensorflow as tf

#Create Streamlit app page
st.set_page_config(page_title="Lyrician", layout="wide",page_icon= "random", initial_sidebar_state="expanded")

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_text()
    return content

bucket_name = "big-data-lyrician"
file_path = "filtered_track_df.csv"

content = read_file(bucket_name, file_path)
with open("checkpoint.csv","w") as file:
    file.write(content + "\n")

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("checkpoint.csv")
    df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
#audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

def page():
    title = "Lyrician"
    st.title(title)
    st.header('Song Recommendation & Lyric Generation Engine')

    st.write("Welcome to our webapp which helps musicians choose out different inspirations, based on the song features. Also, we helps generate fun lyrics with our AI model! :)")
    st.markdown("##")

    with st.container():
        col1, col2,col3,col4 = st.columns((2,0.5,0.5,0.5))
        with col3:
            st.markdown("***Specify the genre:***")
            genre = st.radio(
                "",
                genre_names, index=genre_names.index("Hip Hop"))
        with col1:
            st.markdown("***Choose the year range:***")
            start_year, end_year = st.slider(
                'Select the year range',
                1990, 2019, (2006, 2008)
            )

            st.markdown("***Specify relevant features in this dropdown box:***")
            options = st.multiselect('Relevant features:', options=['acousticness', 'danceability', 'energy','instrumentalness','valence','tempo','liveness','loudness','popularity','speechiness'],
                                        default=['acousticness', 'danceability', 'energy','instrumentalness', 'valence', 'tempo'])

            st.markdown("***Specify the customized song features:***")        
            test_feat=[]
            if "acousticness" in options:
                            acousticness = st.slider(
                            'Acousticness',
                            0.0, 1.0, 0.4)
                            test_feat.append(acousticness)
            if "danceability" in options:
                            danceability = st.slider(
                                'Danceability',
                                0.0, 1.0, 0.7)
                            test_feat.append(danceability)
            if "energy" in options:
                            energy = st.slider(
                                'Energy',
                                0.0, 1.0, 0.5)
                            test_feat.append(energy)
            if "instrumentalness" in options:
                            instrumentalness = st.slider(
                                'Instrumentalness',
                                0.0, 1.0, 0.4)
                            test_feat.append(instrumentalness)
            if "valence" in options:
                            valence = st.slider(
                                'Valence',
                                0.0, 1.0, 0.45)
                            test_feat.append(valence)
            if "tempo" in options:
                            tempo = st.slider(
                                'Tempo',
                                0.0, 244.04, 118.0)
                            test_feat.append(tempo)
            if "liveness" in options:
                            liveness = st.slider(
                                'Liveness',
                                0.0, 1.0, 0.4)
                            test_feat.append(liveness)
            if "loudness" in options:
                            loudness = st.slider(
                                'Loudness',
                                -60.0, -30.0, -0.33)
                            test_feat.append(loudness)
            if "popularity" in options:
                            popularity = st.slider(
                                'Popularity',
                                0.0, 97.0, 48.0)
                            test_feat.append(popularity)
            if "speechiness" in options:
                            speechiness = st.slider(
                                'Speechiness',
                                0.0, 0.941, 0.44)
                            test_feat.append(speechiness)
            

    tracks_per_page = 6
    #test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    audio_feats = []
    for i in options:
        audio_feats.append(i)

    def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
        genre = genre.lower()
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
        genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

        neigh = NearestNeighbors()
        neigh.fit(genre_data[audio_feats].to_numpy())

        n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(audio_feats), return_distance=False)[0]

        uris = genre_data.iloc[n_neighbors]["uri"].tolist()
        audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
        return uris, audios

    uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)

    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
    
    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([0.25,2,0.5,2,0.25])
        if st.button("Show More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i%2==0:
                    with col2:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("Feature radar charts"):
                            df = pd.DataFrame(dict(
                            r=audio,
                            theta=audio_feats))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)
            
                else:
                    with col4:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("Feature radar charts"):
                            df = pd.DataFrame(dict(
                                r=audio,
                                theta=audio_feats))
                            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

        else:
            st.write("No songs left to show")

#page()

def lyr_gen():
    text = open('./Lyrics_txt/text_electronic.txt', 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    def generate_text(model, start_string,t):
    # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 500

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperature results in more predictable text.
        # Higher temperature results in more surprising text.
        # Experiment to find the best setting.
        temperature = t

        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # Pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    input = st.text_input("I would love my lyric to start with:","i just want")
    temp = st.number_input("... with the temperature:",0.3)


    a = keras.models.load_model("./Trained_models/electronic_model.h5")
    st.text(generate_text(a, start_string=input,t=temp))



#Create sidebar
st.sidebar.title("Navigation")
nav = st.sidebar.radio('Pages', options = ["Song features","Lyric generation"])
if nav == "Song features":
    page()
elif nav == "Lyric generation":
    lyr_gen()