from transformers import TransfoXLTokenizer, TransfoXLModel

# Load the tokenizer and model
tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLModel.from_pretrained("transfo-xl-wt103")

# Generate some text
prompt = "The quick brown fox jumps"

prompt = "a smiling woman and a man holding a baby in the street. a large brick building with three windows and a white building behind it.a woman and a little girl standing together near a river with a hat behind them.a lot of houses and boats next to a canal with houses behind ita lot of people standing on a boat with a red tray on the river. a large boat with people riding on the river with buildings behind it.a couple of young children riding on a boat with a person in the canal.a stone building with a wooden bench next to the stone wall.a line of buildings and people riding on a canal next to a canal.a stone building with a metal pole and a door next to the stone wall.a large stone bridge with a person on the boat in the river.a brick building with a red door and a sign next to the water. a lot of boats sitting next to a brick building with a pile of debris.a lot of boats sitting next to a stone house with a blue bag.a woman and a man eating a piece of pizza near the canal.a black background of a black background with a black background"

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
