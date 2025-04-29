from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device('mps')
# === Configuration ===
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Replace with your local path or HF model ID if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(device).eval()

# === Function to Build Prompt ===
def build_prompt(triples):
    examples = """

Given the question 'which is the most relevant book written by achebe?', Choose the most relevant among the following triples:
""" + "\n".join(f"({s}, {p}, {o})" for s, p, o in triples) + "\n\nOutput:"
    return examples.strip()

# === Your Triples to Verbalize ===
triples = [
  [
    "Chinua Achebe",
    "wrote",
    "The Education of a British-Protected Child: Essays"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Things Fall Apart Sparknotes Literature Guide"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "The Drum"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Girls at War and Other Stories"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "A Man of the People"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "No Longer at Ease"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Things Fall Apart"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Chinua Achebe Reading Anthills of the Savannah and Arrow of God"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "A paz dura pouco"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Arrow of God (The African Trilogy, #3)"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Home and Exile"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "How the Leopard Got His Claws"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Things Fall Apart with Connections"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Things Fall Apart, No Longer at Ease, Anthills of the Savannah"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Morning Yet on Creation Day: Essays"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Beware Soul Brother: Poems"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Conversations with Chinua Achebe"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Hopes and Impediments: Selected Essays"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Chike and the River"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "The Flute: A Children's Story"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Things Fall Apart with related readings"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "The Trouble with Nigeria"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "A Summary of Things Fall Apart"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Africa's Tarnished Name"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "Things Fall Apart (The African Trilogy, #1)"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "The University and the Leadership Factor in Nigerian Politics"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "No Longer at Ease (The African Trilogy, #2)"
  ],
  [
    "Chinua Achebe",
    "wrote",
    "The African Trilogy"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Penguin"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Univ. Press of Mississippi"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "National Geographic Books"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "W. W. Norton"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Oxford University Press"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Penguin UK"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Everbind"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Penguin Classics"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "London : Heinemann"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Heinemann"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Anchor Canada"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Editora Companhia das Letras"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Penguin Books Limited"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Holt McDougal"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Everyman's Library"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Heinemann Educational Publishers"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "London : Heinemann Educational"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "East African Publishers"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "SparkNotes"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Spark Notes"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Heinemann International Incorporated"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "MacMillan"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Abic Books"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "Fourth Dimention Publishing Company Limited"
  ],
  [
    "Chinua Achebe",
    "has publisher",
    "York Notes"
  ],
  [
    "Chinua Achebe",
    "citizen of",
    "Nigeria"
  ],
  [
    "Chinua Achebe",
    "birth year",
    "1930"
  ],
  [
    "Chinua Achebe",
    "country of birth",
    "Nigeria"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "africa"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "igbo (african people)"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "juvenile nonfiction"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "african literature"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "christianity"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "english language"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "history"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "short stories, african (english)"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "nigeria"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "city and town life"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "national characteristics, nigerian"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "english fiction"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "british"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "young adult fiction"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "education"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "african fiction (english)"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "children's literature"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "achebe, chinua"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "literary collections"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "africa, west"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "juvenile fiction"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "fiction"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "elections"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "biography & autobiography"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "children's stories"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "children's stories, nigerian (english)"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "study aids"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "interviews"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "african fiction"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "readers (secondary)"
  ],
  [
    "Chinua Achebe",
    "has topic",
    "baseball players"
  ]
]

# === Create Prompt ===
prompt = build_prompt(triples[:8])
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === Generate Output ===
print("✨ Generating verbalization...")
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

# === Extract and Print Result ===
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
verbalization = generated_text.split("Verbalization:")[-1].strip()

print("\n Verbalization:")
print(verbalization)
