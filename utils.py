# load and return the token string from a file 

def load_token():
    path = "/data1/malto/unlearning_llm/token.txt"
    with open(path, 'r') as f:
        return f.read().strip()