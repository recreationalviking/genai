from gpu_memory_fitter import GPUMemoryFitter
from llama_cpp import Llama

model_size = 70
model_multiplier = .5
gpu_manager = GPUMemoryFitter()

# Get GPU info
tensor_split = gpu_manager.create_llama_cpp_tensor_split(model_size, model_multiplier)
print(f"Device memory map for the model with {model_size}GB:", tensor_split)

# Instantiate model
llm = Llama(".models/TheBloke-llama-2-70b.q4_k_m/llama-2-70b.Q4_K_M.gguf", n_gpu_layers=1000, n_threads=8, device_map = "auto", tensor_split = tensor_split['tensor'], main_gpu=tensor_split['main_gpu'])
output = llm(
      "Q: What is the answer to the ultimate question about life, the universe, and everything? A: The answer is ", # Prompt
      max_tokens=300, # Generate up to X tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)

print(output["choices"][0]["text"].strip())

print("done.")
exit(0)
