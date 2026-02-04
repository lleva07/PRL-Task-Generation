from langchain_community.llms import Ollama

print("Connecting to Ollama...")
llm = Ollama(model="qwen2.5:1.5b") # Make sure this matches your pulled model
response = llm.invoke("Say 'System Online' if you can hear me.")
print(f"Ollama says: {response}")