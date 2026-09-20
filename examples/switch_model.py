"""The same client drives every hosted model listed in codeformer_api.MODELS."""
from codeformer_api import Client, MODELS

client = Client()
for slug, info in MODELS.items():
    print(slug, "->", info["category"], "required:", info["required"])
# pick one explicitly
output = client.run({"img": "https://example.com/input.png"}, model="tencentarc/gfpgan")
print(output)
