import pickle

import pandas as pd
import streamlit as st
import requests


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=03e43f4712813d897dcd9b5462e31b9b&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


st.header('Movie Recommender System')
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)


selected_movie = st.selectbox(

    'How would you like to be contacted?',
    movies['title'].values)

if st.button('Show Recommendation'):
    recommended_movie_name, recommended_movie_poster = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_name[0])
        st.image(recommended_movie_poster[0])
    with col2:
        st.text(recommended_movie_name[1])
        st.image(recommended_movie_poster[1])

    with col3:
        st.text(recommended_movie_name[2])
        st.image(recommended_movie_poster[2])
    with col4:
        st.text(recommended_movie_name[3])
        st.image(recommended_movie_poster[3])
    with col5:
        st.text(recommended_movie_name[4])
        st.image(recommended_movie_poster[4])
