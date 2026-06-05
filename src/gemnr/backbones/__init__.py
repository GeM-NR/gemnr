SUPPORTED_BACKBONES = [
    "brushnet",
    "flux2_klein",
    "flux2_klein_4B",
    "qwen",
    "qwen_2509",
    "qwen_2511",
    "edicho",
]


class ImageEditor:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, inputs, prompt):
        return self.run_pipe(inputs, prompt)

    def run_pipe(self, inputs, prompt):
        raise NotImplementedError
