import subprocess
import json

# Function to load commands from JSON
def load_commands(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['commands']

if __name__ == "__main__":
    # Load commands from the JSON file
    commands = load_commands('commands.json')

    # Run each command in sequence
    for command in commands:
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Command {' '.join(command)} failed with exit code {result.returncode}")
            break

# example of commands.json
#{"commands": [["python", "your_script.py", "--config", "config1.json"], ["python", "another_script.py"], ["python", "third_script.py"]]}