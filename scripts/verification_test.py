from multimodal_utils import OllamaInterface
import time

def test_ollama():
    print("Initializing Ollama Interface...")
    ollama = OllamaInterface(model_name="qwen2.5vl")
    
    print("Testing Text Chat...")
    start = time.time()
    response = ollama.chat([{"role": "user", "content": "Hello, are you ready?"}])
    print(f"Response: {response}")
    print(f"Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_ollama()
