from nlpython.llm import MODEL_TO_API, SUPPORTED_MODELS


class NLPDebugger:
    _system_message = (
        f"You are a python code debugger. You take as input the code and the "
        f"error message and you must return a new version of the code that "
        f"fixes the error.\nFormat your answer as plain code and mark the "
        f"beginning of the code with <code> and the end with </code>."
    )

    def __init__(
            self,
            model_name: str = "gpt-3.5-turbo",
            max_tokens: int = 1024,
            temperature: float = 0.7,
            max_iterations: int = 10,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} is not supported. "
                f"Supported models are: {SUPPORTED_MODELS}"
            )
        self._model_name = model_name
        self._llm = MODEL_TO_API[model_name]()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_iterations = max_iterations

    def debug(self, code: str):
        end = False
        iteration = 0
        while not end:
            if iteration > self._max_iterations:
                raise Exception(
                    f"Max iterations reached. "
                    f"Last code generated:\n{code}"
                )
            try:
                exec(code)
                end = True
            except Exception as e:
                message = (
                    f"While running the code below:\n{code} "
                    f"the following exception was "
                    f"raised:\n{e}\nPlease fix the code!"
                )
                new_code = self._llm.run(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": self._system_message},
                        {"role": "user", "content": message},
                    ],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                if "<code>" in new_code and "</code>" in new_code:
                    new_code = new_code.split("<code>")[1].split("</code>")[0].strip()
                else:
                    new_code = code
                code = new_code
                iteration += 1
        return code
