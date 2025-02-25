# import os # Optional for faster downloading
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from dotenv import load_dotenv
import os
from huggingface_hub import snapshot_download

# Load environment variables from .env file
load_dotenv()

def donwload_unsloth():
    # Set proxy environment variables
    if os.getenv('HTTP_PROXY'):
        os.environ['http_proxy'] = os.getenv('HTTP_PROXY')
    if os.getenv('HTTPS_PROXY'):
        os.environ['https_proxy'] = os.getenv('HTTPS_PROXY')
    
    snapshot_download(
        repo_id = "unsloth/DeepSeek-R1-GGUF",
        local_dir = "DeepSeek-R1-GGUF",
        allow_patterns = ["*UD-IQ1_S*"], # Select quant type UD-IQ1_S for 1.58bit
    )

if __name__ == "__main__": 
    donwload_unsloth()
