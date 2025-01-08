# HOWTO Interactive Mode

## Start server

### UniCRS

Start the server with the following command (RedDial dataset):

```bash
python -m script.serve_model --crs_model unicrs --kg_dataset redial --model microsoft/DialoGPT-small --rec_model data/models/unicrs_rec_redial/ --conv_model data/models/unicrs_conv_redial/ --context_max_length 128 --entity_max_length 43 --tokenizer_path microsoft/DialoGPT-small --text_tokenizer_path roberta-base --resp_max_length 128 --text_encoder roberta-base --debug
```

### BARCOR

Start the server with the following command (RedDial dataset):

```bash
python -m script.serve_model --crs_model barcor --kg_dataset redial  --context_max_length 200 --rec_model data/models/barcor_rec_redial/ --conv_model data/models/barcor_conv_redial/ --tokenizer_path facebook/bart-base --resp_max_length 128 --debug
```

### KBRD

Start the server with the following command (RedDial dataset):

```bash
python -m script.serve_model --crs_model kbrd --kg_dataset redial --hidden_size 128 --entity_hidden_size 128 --num_bases 8  --context_max_length 200 --entity_max_length 32 --rec_model data/models/kbrd_rec_redial/ --conv_model data/models/kbrd_conv_redial/ --tokenizer_path facebook/bart-base --encoder_layers 2 --decoder_layers 2 --attn_head 2 --text_hidden_size 300 --resp_max_length 128
```

### ChatGPT

Start the server with the following command (RedDial dataset):

```bash
python -m script.serve_model --api_key {API_KEY} --kg_dataset redial --crs_model chatgpt
```

Note that the item embeddings should be computed before starting the server and stored in the `data/embed_items/{kg_dataset}` folder.

## Communicate with the server

Test in the terminal with the following command:

```python
import requests

url = "http://127.0.0.1:5005/"
s = requests.Session() 

context = []
data = {
    "context": context,
    "message": "Hi I am looking for a movie like Super Troopers (2001)",
}

response = s.post(url, json=data)
print(response.status_code)
print(response.json())

response = response.json()

context += ["Hi I am looking for a movie like Super Troopers (2001)", response.get("response")]
data = {
    "context": context,
    "message": "I love action movies",
}

response = s.post(url, json=data)
```

## Start Streamlit app

A Streamlit is available to collect conversational data from users. The idea is to put two models in competition and ask the best model based on the user's feedback.

```bash
python -m streamlit run crs_arena/arena.py
```

The configuration of the CRSs are in the `data/arena/crs_config/` folder. The available models with their associated configuration are defined in `CRS_MODELS` in `crs_arena/battle_manager.py`.

The conversation logs are stored in the `data/arena/conversation_logs/` folder. The votes are registered in the `data/arena/vote.db` SQLite database.
