import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import altair as alt
import os

import base64
import os
import json
import pickle
import uuid
import re


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 

            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


# Set wide page mode
st.set_page_config(layout="wide")

#Read image and decorate with streamlit cache function to avoid reloading each time script executes:
@st.cache
def load_grid_image():
    grid = plt.imread('grid.jpg')
    return grid

grid = load_grid_image()

   
# Now start building up the display with some titles.
st.title('Energy Busters -  Energy Transition Predictor')

# Radio selector om een scherm te kiezen
screen = st.sidebar.radio('Selecteer het scherm', ['Project plaatsing'])

screen1=False
screen2=False
screen3=False

if screen == 'Project plaatsing':
    screen1=True
elif screen == 'Scherm 2':
    screen2=True
elif screen == 'Scherm 3':   
    screen3=True

if screen1:

    # Display some headers
    st.header('Screen 1: Project plaatsing')
    st.sidebar.header('Project plaatsen')

    # Define 2 columns with a dummy 'spacer column for better layout (column object is in beta) 
    c1, dum, c2 = st.beta_columns((10,1,10))

    c1.subheader('Grid kaart')
    c2.subheader('Details')

    # st.write(f'Current working dir {os.getcwd()}')

    try:
        projects = pd.read_csv('/Users/jandeknatel/OneDrive/Documents/GitHub/Energy_Busters/projects.csv')
        num_projects = projects.shape()[1]
        # st.write(f' success {num_projects}')
    except:
        projects = pd.DataFrame(columns=['x','y','type','vermogen'])
        num_projects = 0
        # st.write(num_projects)

    x1 = st.sidebar.slider('Selecteer de X-positie van het project', 0, 513, 250, 1)
    y1_min = 0
    y1_max = 540
   
    y2 = st.sidebar.slider('Selecteer de Y-positie van het project', 0, 540, 250, 2)
    x2_min = 0
    x2_max = 513


    fig, ax = plt.subplots()
    ax.imshow(grid, extent=[0, 513, 0, 540])

    x = [x1,x1]
    y = [y1_min, y1_max]
    ax.plot(x, y, '--', linewidth=1, color='blue')

    x = [x2_min,x2_max]
    y = [y2, y2]
    ax.plot(x, y, '--', linewidth=1, color='blue')

    # Radio selector om een scherm te kiezen
    type = st.sidebar.radio('Selecteer type project', ['Wind', 'Zon'])
    power = st.sidebar.slider('Selecteer vermogen van het project in Megawatt', 0, 100, 50, 5)

    if type == 'Wind':
        color='c'
    elif type == 'Zon':
        color='r'
    else:
        Print(f'Invalid choice of type {type}')

    cir = plt.Circle((x1, y2), power/5, color=color,fill=True)
    ax.add_patch(cir)

    c1.pyplot(fig)

    projects.loc[num_projects] = [x1, y2, type, power]

    c2.dataframe(projects)

    download_button_str = download_button(projects, 'projects.json', 'Click here to download dataframe', pickle_it=False)
    c2.markdown(download_button_str, unsafe_allow_html=True)

