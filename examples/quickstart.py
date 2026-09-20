        """Minimal CodeFormer example: create one prediction and print the output URL(s)."""
        import codeformer_api

        output = codeformer_api.run({
    "image": "https://example.com/input.png"
})
        print(output)
