import os
import io
import sys
import json
import requests
from dotenv import load_dotenv
import anthropic
import gradio as gr
import subprocess
import platform
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer

load_dotenv(override=True)
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

claude = anthropic.Anthropic()
CLAUDE_MODEL = "claude-sonnet-4-20250514"
code_qwen = "Qwen/CodeQwen1.5-7B-Chat"
CODE_QWEN_URL = "https://zzgg5rv8a6d0l4xp.us-east4.gcp.endpoints.huggingface.cloud"

# Compiler detection constants
VISUAL_STUDIO_2022_TOOLS = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\VsDevCmd.bat"
VISUAL_STUDIO_2019_TOOLS = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\Common7\\Tools\\VsDevCmd.bat"

simple_cpp = """
#include <iostream>

int main() {
    std::cout << "Hello";
    return 0;
}
"""

def run_cmd(command_to_run):
    try:
        run_result = subprocess.run(command_to_run, check=True, text=True, capture_output=True)
        return run_result.stdout if run_result.stdout else "SUCCESS"
    except:
        return ""

def c_compiler_cmd(filename_base):
    my_platform = platform.system()
    my_compiler = []

    try:
        with open("simple.cpp", "w") as f:
            f.write(simple_cpp)
            
        if my_platform == "Windows":
            if os.path.isfile(VISUAL_STUDIO_2022_TOOLS):
                if os.path.isfile("./simple.exe"):
                    os.remove("./simple.exe")
                compile_cmd = ["cmd", "/c", VISUAL_STUDIO_2022_TOOLS, "&", "cl", "simple.cpp"]
                if run_cmd(compile_cmd):
                    if run_cmd(["./simple.exe"]) == "Hello":
                        my_compiler = ["Windows", "Visual Studio 2022", ["cmd", "/c", VISUAL_STUDIO_2022_TOOLS, "&", "cl", f"{filename_base}.cpp"]]
        
            if not my_compiler:
                if os.path.isfile(VISUAL_STUDIO_2019_TOOLS):
                    if os.path.isfile("./simple.exe"):
                        os.remove("./simple.exe")
                    compile_cmd = ["cmd", "/c", VISUAL_STUDIO_2019_TOOLS, "&", "cl", "simple.cpp"]
                    if run_cmd(compile_cmd):
                        if run_cmd(["./simple.exe"]) == "Hello":
                            my_compiler = ["Windows", "Visual Studio 2019", ["cmd", "/c", VISUAL_STUDIO_2019_TOOLS, "&", "cl", f"{filename_base}.cpp"]]
    
            if not my_compiler:
                my_compiler=[my_platform, "Unavailable", []]
                
        elif my_platform == "Linux":
            if os.path.isfile("./simple"):
                os.remove("./simple")
            compile_cmd = ["g++", "simple.cpp", "-o", "simple"]
            if run_cmd(compile_cmd):
                if run_cmd(["./simple"]) == "Hello":
                    my_compiler = ["Linux", "GCC (g++)", ["g++", f"{filename_base}.cpp", "-o", f"{filename_base}" ]]
    
            if not my_compiler:
                if os.path.isfile("./simple"):
                    os.remove("./simple")
                compile_cmd = ["clang++", "simple.cpp", "-o", "simple"]
                if run_cmd(compile_cmd):
                    if run_cmd(["./simple"]) == "Hello":
                        my_compiler = ["Linux", "Clang++", ["clang++", f"{filename_base}.cpp", "-o", f"{filename_base}"]]
        
            if not my_compiler:
                my_compiler=[my_platform, "Unavailable", []]
    
        elif my_platform == "Darwin":
            # Try multiple clang++ configurations for macOS (M1/M2 Apple Silicon)
            test_configs = [
                # Try conda clang++ first (if available) - basic
                ["clang++", "-std=c++17", "-o", "simple", "simple.cpp"],
                # Try conda clang++ with optimization
                ["clang++", "-O3", "-std=c++17", "-o", "simple", "simple.cpp"],
                # Try system clang++ with basic flags
                ["/usr/bin/clang++", "-std=c++17", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-O3", "-std=c++17", "-o", "simple", "simple.cpp"],
                # Try with explicit SDK path (latest)
                ["/usr/bin/clang++", "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk", "-std=c++17", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk", "-O3", "-std=c++17", "-o", "simple", "simple.cpp"],
                # Try with specific SDK versions for Sequoia
                ["/usr/bin/clang++", "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk", "-O3", "-std=c++17", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk", "-O3", "-std=c++17", "-o", "simple", "simple.cpp"],
                # Try M2-specific optimizations (armv8.6-a for M2)
                ["clang++", "-O3", "-std=c++17", "-march=armv8.6-a", "-mtune=apple-m2", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-O3", "-std=c++17", "-march=armv8.6-a", "-mtune=apple-m2", "-o", "simple", "simple.cpp"],
                # Try M1 optimizations (fallback)
                ["clang++", "-O3", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-O3", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-o", "simple", "simple.cpp"],
                # Try original flags from your code
                ["clang++", "-O3", "-std=c++17", "-march=armv8.3-a", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-O3", "-std=c++17", "-march=armv8.3-a", "-o", "simple", "simple.cpp"],
                # Try with Xcode SDK and M2 flags
                ["/usr/bin/clang++", "-isysroot", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk", "-O3", "-std=c++17", "-march=armv8.6-a", "-o", "simple", "simple.cpp"],
                # Fallback: very basic compilation
                ["clang++", "-o", "simple", "simple.cpp"],
                ["/usr/bin/clang++", "-o", "simple", "simple.cpp"]
            ]
            
            for compile_cmd in test_configs:
                if os.path.isfile("./simple"):
                    os.remove("./simple")
                
                if run_cmd(compile_cmd):
                    if run_cmd(["./simple"]) == "Hello":
                        # Create the actual command for the filename
                        actual_cmd = compile_cmd.copy()
                        # Replace "simple.cpp" with the actual filename
                        for i, arg in enumerate(actual_cmd):
                            if arg == "simple.cpp":
                                actual_cmd[i] = f"{filename_base}.cpp"
                            elif arg == "simple":
                                actual_cmd[i] = filename_base
                        
                        compiler_name = "Clang++"
                        if "-march=armv8.6-a" in compile_cmd:
                            compiler_name = "Clang++ (M2 Optimized)"
                        elif "-march=armv8.5-a" in compile_cmd:
                            compiler_name = "Clang++ (M1 Optimized)"
                        elif "-march=armv8.3-a" in compile_cmd:
                            compiler_name = "Clang++ (ARM Optimized)"
                        elif "/usr/bin/clang++" in compile_cmd:
                            compiler_name = "System Clang++"
                        elif "isysroot" in " ".join(compile_cmd):
                            compiler_name = "Clang++ (SDK)"
                        elif "-O3" in compile_cmd:
                            compiler_name = "Clang++ (Optimized)"
                            
                        my_compiler = ["Macintosh", compiler_name, actual_cmd]
                        break
    
            if not my_compiler:
                my_compiler=[my_platform, "Unavailable", []]
    except:
        my_compiler=[my_platform, "Unavailable", []]
        
    # Clean up test file
    try:
        if os.path.isfile("simple.cpp"):
            os.remove("simple.cpp")
        if os.path.isfile("simple") or os.path.isfile("./simple"):
            os.remove("simple")
    except:
        pass
        
    if my_compiler:
        return my_compiler
    else:
        return ["Unknown", "Unavailable", []]

# Detect compiler at startup
compiler_cmd = c_compiler_cmd("optimized")
print(f"Detected compiler: {compiler_cmd[0]} - {compiler_cmd[1]}")

system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time. Keep implementations of random number generators identical so that results match exactly. Do not add any additional explanation to your code"

def user_prompt_for(python):
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python
    return user_prompt

def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

# write to a file called optimized.cpp

def write_output(cpp):
    code = cpp.replace("```cpp","").replace("```","")
    with open("optimized.cpp", "w") as f:
        f.write(code)

def optimize_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            print(text, end="", flush=True)
    write_output(reply)

python_hard = """# Be careful to support large number sizes

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value
        
def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def total_max_subarray_sum(n, initial_seed, min_val, max_val):
    total_sum = 0
    lcg_gen = lcg(initial_seed)
    for _ in range(20):
        seed = next(lcg_gen)
        total_sum += max_subarray_sum(n, seed, min_val, max_val)
    return total_sum

# Parameters
n = 10000         # Number of random numbers
initial_seed = 42 # Initial seed for the LCG
min_val = -10     # Minimum value of random numbers
max_val = 10      # Maximum value of random numbers

# Timing the function
import time
start_time = time.time()
result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
end_time = time.time()

print("Total Maximum Subarray Sum (20 runs):", result)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
"""

def stream_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```cpp\n','').replace('```','')

def stream_code_qwen(python):
    tokenizer = AutoTokenizer.from_pretrained(code_qwen)
    messages = messages_for(python)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    client = InferenceClient(CODE_QWEN_URL, token=hf_token)
    stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
    result = ""
    for r in stream:
        result += r.token.text
        # Clean up Qwen-specific tokens and artifacts
        cleaned_result = result
        
        # Remove model control tokens (Qwen specific)
        cleaned_result = cleaned_result.replace('<|im_start|>', '')
        cleaned_result = cleaned_result.replace('<|im_end|>', '')
        cleaned_result = cleaned_result.replace('<|endoftext|>', '')
        cleaned_result = cleaned_result.replace('<|user|>', '')
        cleaned_result = cleaned_result.replace('<|assistant|>', '')
        
        # Remove any remaining angle bracket tokens with regex
        import re
        cleaned_result = re.sub(r'<\|[^|]*\|>', '', cleaned_result)
        
        # Remove markdown code blocks if present
        cleaned_result = cleaned_result.replace('```cpp\n', '').replace('```cpp', '').replace('```', '')
        
        # Clean up common formatting issues
        cleaned_result = cleaned_result.strip()
        
        # Remove any text after a common end marker that might indicate model confusion
        if '</code>' in cleaned_result:
            cleaned_result = cleaned_result.split('</code>')[0]
        
        yield cleaned_result

def execute_python(code):
    try:
        output = io.StringIO()
        sys.stdout = output
        exec(code)
    finally:
        sys.stdout = sys.__stdout__
    return output.getvalue()

def execute_cpp(code):
    write_output(code)
    if compiler_cmd[1] == "Unavailable":
        return "Error: No C++ compiler available"
    
    try:
        compile_result = subprocess.run(compiler_cmd[2], check=True, text=True, capture_output=True)
        run_cmd = ["./optimized"]
        run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
        return run_result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred:\n{e.stderr}"
    
css = """
.python {background-color: #306998;}
.cpp {background-color: #050;}
"""

def optimize(python, model):
    if model=="Claude":
        result = stream_claude(python)
    elif model=="CodeQwen":
        result = stream_code_qwen(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far  

with gr.Blocks(css=css) as ui:
    gr.Markdown("## Convert code from Python to C++")
    with gr.Row():
        python = gr.Textbox(label="Python code:", value=python_hard, lines=10)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(["Claude", "CodeQwen"], label="Select model", value="Claude")
        with gr.Column():
            architecture = gr.Radio([compiler_cmd[0]], label="Architecture", interactive=False, value=compiler_cmd[0])
            compiler = gr.Radio([compiler_cmd[1]], label="Compiler", interactive=False, value=compiler_cmd[1])
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Python")
        if compiler_cmd[1] != "Unavailable":
            cpp_run = gr.Button("Run C++")
        else:
            cpp_run = gr.Button("No compiler to run C++", interactive=False)
    with gr.Row():
        python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
        cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

    convert.click(optimize, inputs=[python, model], outputs=[cpp])
    python_run.click(execute_python, inputs=[python], outputs=[python_out])
    cpp_run.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)