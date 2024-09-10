from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
text = "你好1吗?"
# text = "the cat is in the hat"

print(len(text))

unicode_code_points = [ord(char) for char in text]
print(f"unicode_code_points = {unicode_code_points}")
print(f"len unicode_code_points = {len(unicode_code_points)} \n")

byte_sequence = text.encode("utf-8")
print(f"byte_sequence = {byte_sequence}")
print(f"len byte_sequence = {len(byte_sequence)}  \n")

tokens = tokenizer.encode(text)
for token in tokens:
    string = tokenizer.convert_ids_to_tokens(token)
    print(f" ID: {token}       string: {string}      bytes: {string.encode('utf-8')}")
print(f"tokens = {tokens}")

print(f"tokenizer decode = {tokenizer.decode(tokens)}")

string_tokens = tokenizer.convert_ids_to_tokens(tokens)
text = "".join(string_tokens)
print(f"text = {text}")
for c in text:
    print(f"char = {c}      byte = {bytes([tokenizer.byte_decoder[c]])}")
decoded_manual = bytearray([tokenizer.byte_decoder[c] for c in text]).decode(
    "utf-8", errors="replace"
)
print(f"decoded_manual = {decoded_manual}")
