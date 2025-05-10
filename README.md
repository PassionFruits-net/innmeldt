# Knowledge Bot

## Prerequisites

Fill out .env.template then rename it '.env'

## Setup Instructions

1. **Install Dependencies**  
   First, install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**  
   Open a terminal and run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Run the API Server**  
   In a separate terminal window, run the API server using Uvicorn:
   ```bash
   uvicorn api:app --reload
   ```
