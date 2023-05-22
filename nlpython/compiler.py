import os
from pathlib import Path

from nlpython.converter import NLPConverter
from nlpython.debugger import NLPDebugger


class NLPythonCompiler:
    def __init__(self):
        self._converter = NLPConverter(
            model_name=os.getenv("NLPYTHON_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("NLPYTHON_MAX_TOKENS", 2048)),
            temperature=float(os.getenv("NLPYTHON_TEMPERATURE", 0.7)),
        )
        self._debugger = NLPDebugger(
            model_name=os.getenv("NLPYTHON_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("NLPYTHON_MAX_TOKENS", 2048)),
            temperature=float(os.getenv("NLPYTHON_TEMPERATURE", 0.7)),
            max_iterations=int(os.getenv("NLPYTHON_MAX_ITERATIONS", 10)),
        )

    def compile(self, code: str):
        converted_code = self._converter.convert(code)
        final_code = self._debugger.debug(converted_code)
        try:
            exec(final_code)
        except Exception as e:
            raise Exception(
                f"An exception was raised while running the code:\n{final_code}\n"
                f"The exception was:\n{e}"
            )

        return final_code
    

def compile_file(path_to_code: str) -> str:
    with open(path_to_code, "r") as f:
        code = f.read()
    compiled_code = NLPythonCompiler().compile(code)
    python_code_path = Path(path_to_code).parent / "_nlp_compiled.py"
    with open(python_code_path, "w") as f:
        f.write(compiled_code)
    return str(python_code_path)
