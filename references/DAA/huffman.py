# Huffman Encoding using Greedy Method

text = input("Enter a string: ")

# Count frequency of characters
freq = {}
for ch in text:
    freq[ch] = freq.get(ch, 0) + 1

# Create nodes
nodes = [[ch, f] for ch, f in freq.items()]

# Build Huffman Tree
while len(nodes) > 1:

    nodes.sort(key=lambda x: x[1])

    left = nodes.pop(0)
    right = nodes.pop(0)

    new_node = [None, left[1] + right[1], left, right]
    nodes.append(new_node)

# Generate Huffman Codes
codes = {}

def generate(node, code=""):
    if node[0] is not None:
        codes[node[0]] = code if code else "0"
        return

    generate(node[2], code + "0")
    generate(node[3], code + "1")

# Display codes and encoded string
if nodes:
    generate(nodes[0])

    print("\nHuffman Codes:")
    for ch, code in codes.items():
        print(ch, ":", code)

    encoded = ""
    for ch in text:
        encoded += codes[ch]

    print("Encoded String:", encoded)

else:
    print("Please enter a non-empty string.")