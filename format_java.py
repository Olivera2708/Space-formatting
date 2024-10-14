import subprocess
import tempfile
from pathlib import Path


def wrap_in_class(snippet):
    wrapped_code = f"public class Temp {{\n{snippet}\n}}"
    return wrapped_code

def format_java_code(java_code):
    java_code = wrap_in_class(java_code)
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".java") as temp_file:
            temp_file.write(java_code)
            temp_file_path = temp_file.name
        
        try:
            google_java_format_jar = 'google-java-format-1.24.0-all-deps.jar'
            command = ['java', '-jar', google_java_format_jar, '--replace', temp_file_path]
            subprocess.run(command, check=True)
            with open(temp_file_path, 'r') as formatted_file:
                formatted_code = formatted_file.read()

            return formatted_code.replace("public class Temp {", "").replace("}", "").strip()

        except subprocess.CalledProcessError as e:
            print(f"Error during formatting: {e}")
            return None
        
    except Exception as e:
        print(f"Unexpected error: {e}. Skipping this snippet and continuing.")
        return None
    
    finally:
        Path(temp_file_path).unlink()