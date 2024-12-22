import spacy

# Load a larger pre-trained spaCy model
nlp = spacy.load("en_core_web_md")

# Add custom rules using EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "ORG", "pattern": "spaCy"},
    {"label": "GPE", "pattern": "San Francisco"}
]
ruler.add_patterns(patterns)

# Input sentences
text = "Peter and Jim go to school by bus. Jim takes Sam to the library for their school project. After a long workday, Jim and Sam meet Peter at the restaurant in San Francisco."

# Process text
doc = nlp(text)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Print recognized entities
print("Entities Identified:")
for entity in entities:
    print(f" - {entity[0]}: {entity[1]}")

# Extract actions and associate with entities
print("\nActivities Identified:")
for token in doc:
    if token.pos_ == "VERB":
        subject = [child.text for child in token.children if child.dep_ == "nsubj"]
        obj = [child.text for child in token.children if child.dep_ == "dobj"]
        print(f" - Action: {token.text}, Subject: {', '.join(subject)}, Object: {', '.join(obj)}")
