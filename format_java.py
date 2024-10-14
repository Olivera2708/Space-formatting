import subprocess
import tempfile
from pathlib import Path
import re


def create_vocab(tokens, vocab_stoi, vocab_itos):
    for current_token, current_type in tokens:
        if current_type == 6:
            if '\t' in current_token:
                vocab_stoi["<TAB>"] = len(vocab_stoi)
                vocab_itos[len(vocab_itos)] = "<TAB>"
            elif '\n' in current_token:
                vocab_stoi["<NEWLINE>"] = len(vocab_stoi)
                vocab_itos[len(vocab_itos)] = "<NEWLINE>"
            else:
                vocab_stoi["<SPACE>"] = len(vocab_stoi)
                vocab_itos[len(vocab_itos)] = "<SPACE>"

        elif current_token not in vocab_stoi:
            vocab_stoi[current_token] = len(vocab_stoi)
            vocab_itos[len(vocab_itos)] = current_token
    
    return vocab_stoi, vocab_itos


def tokenize_and_classify(code):
    token_patterns = {
        "keyword": r'\b(public|private|protected|static|final|transient|volatile|abstract|synchronized|native|strictfp|interface|implements|extends|super|this|class|enum|package|import|return|void|if|else|for|while|do|switch|case|default|break|continue|try|catch|finally|throw|throws|assert|instanceof|new|instanceof|const|goto|boolean|byte|char|short|int|long|float|double)\b',
        "identifier": r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
        "operator": r'[+\-*/=<>!&|%^]',
        "punctuation": r'[(){}[\];,.]',
        "literal": r'\b\d+(\.\d+)?\b',
        "whitespace": r'\s+',
    }

    combined_pattern = '|'.join(f'(?P<{key}>{value})' for key, value in token_patterns.items())    
    tokens = []
    for match in re.finditer(combined_pattern, code):
        for token_type, token_value in match.groupdict().items():
            if token_value:
                if token_type == 'keyword':
                    tokens.append((token_value, 1))
                elif token_type == 'identifier':
                    tokens.append((token_value, 2))
                elif token_type == 'operator':
                    tokens.append((token_value, 3))
                elif token_type == 'punctuation':
                    tokens.append((token_value, 4))
                elif token_type == 'literal':
                    tokens.append((token_value, 5))
                elif token_type == 'whitespace':
                    tokens.append((token_value, 6))

    return tokens

def tokenize_and_create_input_output(tokens, vocab_stoi):
    input = []
    output = []
    i = 0
    
    while i < len(tokens) - 1:
        current_token, current_type = tokens[i]
        next_token, next_type = tokens[i + 1]

        space_type = "<SPACE>"
        space_count = 0

        if current_type == 6:
            i+=1
            continue

        try:
            if next_type == 6:
                whitespace = next_token
                next_token, next_type = tokens[i + 2]

                if '\t' in whitespace:
                    space_type = '<TAB>'
                    space_count = 1
                elif '\n' in whitespace:
                    space_type = '<NEWLINE>'
                    space_count = 1
                else:
                    space_type = '<SPACE>'
                    space_count = len(whitespace)
            
                input.append((vocab_stoi[current_token], current_type, vocab_stoi[next_token], next_type))
                output.append((vocab_stoi[space_type], space_count))
        except Exception as e:
            pass
        i += 1

    return input, output

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