import os
import yaml
import streamlit as st
from dotenv import load_dotenv
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from datetime import datetime

def handle_authentication():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        auto_hash=True
    )

    try:
        authenticator.login("sidebar")
    except Exception as e:
        st.error(f"Authentication error: {e}")

    if st.session_state["authentication_status"]:
        session_id = f'{st.session_state["username"]}/{datetime.now().strftime("%Y%m%d%H")}'
        st.session_state["user_sessionid"] = session_id
        authenticator.logout(location="sidebar")
        st.write(f"Welcome *{st.session_state['name']}*!")
        load_dotenv()
        st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        return True
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        return False
    else:
        st.warning("Please log in to continue.")
        return False