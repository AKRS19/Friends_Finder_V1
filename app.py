import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # nowa biblioteka do wykresów wskaźnikowych

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'find_friends_cluster_names_and_descriptions_v1.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

st.title("Friend Finder")

# Sekcja z dynamicznym kolorem napisu
if "interaction" not in st.session_state:
    st.session_state.interaction = False

if st.session_state.interaction:
    text_color = "black"
else:
    text_color = "red"

st.markdown(
    f"""
    <style>
    .dynamic-text {{
        color: {text_color};
        animation: {"none" if text_color == "black" else "pulse 2s infinite"};
    }}
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
        100% {{ opacity: 1; }}
    }}
    </style>
    <p class="dynamic-text">Wprowadź swoje dane na pasku po lewej stronie, aby znaleźć osoby z podobnymi zainteresowaniami!</p>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'], key="age")
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'], key="edu_level")
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'], key="fav_animals")
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'], key="fav_place")
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'], key="gender")

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

    # Zarejestruj interakcję użytkownika
    st.session_state.interaction = True

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# Zmienne do wykresu wskaźnikowego
num_friends = len(all_df[all_df["Cluster"] == predicted_cluster_id])
total_participants = len(all_df)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=num_friends,
    gauge={
        'axis': {'range': [0, total_participants]},
        'bar': {'color': "red"},
    },
    title={'text': "Liczba znajomych"}
))

# Sekcja główna (bez kolumn)
st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
st.metric("Liczba twoich znajomych", num_friends)
st.plotly_chart(fig_gauge, use_container_width=True)

# Histogramy
st.header("Osoby z grupy")

fig_age = px.histogram(
    all_df[all_df["Cluster"] == predicted_cluster_id].sort_values("age"),
    x="age",
    color="age",  # Różne kolory dla każdego słupka
    color_discrete_map={str(age): px.colors.qualitative.Set3[i] for i, age in enumerate(all_df["age"].unique())}
)
fig_age.update_layout(title="Rozkład wieku w grupie", xaxis_title="Wiek", yaxis_title="Liczba osób", showlegend=False)
st.plotly_chart(fig_age, use_container_width=True)

fig_edu = px.histogram(
    all_df[all_df["Cluster"] == predicted_cluster_id],
    x="edu_level",
    color="edu_level",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_edu.update_layout(title="Rozkład wykształcenia w grupie", xaxis_title="Wykształcenie", yaxis_title="Liczba osób", showlegend=False)
st.plotly_chart(fig_edu, use_container_width=True)

fig_animals = px.histogram(
    all_df[all_df["Cluster"] == predicted_cluster_id],
    x="fav_animals",
    color="fav_animals",
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig_animals.update_layout(title="Rozkład ulubionych zwierząt w grupie", xaxis_title="Ulubione zwierzęta", yaxis_title="Liczba osób", showlegend=False)
st.plotly_chart(fig_animals, use_container_width=True)

fig_place = px.histogram(
    all_df[all_df["Cluster"] == predicted_cluster_id],
    x="fav_place",
    color="fav_place",
    color_discrete_sequence=px.colors.qualitative.Vivid
)
fig_place.update_layout(title="Rozkład ulubionych miejsc w grupie", xaxis_title="Ulubione miejsce", yaxis_title="Liczba osób", showlegend=False)
st.plotly_chart(fig_place, use_container_width=True)

fig_gender = px.histogram(
    all_df[all_df["Cluster"] == predicted_cluster_id],
    x="gender",
    color="gender",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_gender.update_layout(title="Rozkład płci w grupie", xaxis_title="Płeć", yaxis_title="Liczba osób", showlegend=False)
st.plotly_chart(fig_gender, use_container_width=True)
