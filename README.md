# End to End Multi AI RAG ChatBot using Langgraph and Astra DB

### Run Experiments folder files if you want in Colab cause it has computation power with GPU

Make sure to add HF_TOKEN, ASTRA_DB credentials Tokens, GROQ_API_KEY token into secrets if running experiments in Google colab

Local run

```
conda create -p venv python==3.12 -y
```

```
pip install -r requirements.txt
```

Make sure to add ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID, GROQ_API_KEY, HF_TOKEN in .env file and uncomment the environ codes in app.py and comment out st.secrets code lines

```
streamlit run app.py
```