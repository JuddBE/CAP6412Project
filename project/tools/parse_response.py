import sys
import os
import re
import pandas as pd

from prompts import Prompts

class CSVModifier:
    def __init__(self, file_path, prompt_idx, output_path=None):
        self.file_path = file_path
        self.text_prompts = Prompts().GetPromptList(prompt_idx)
        self.prompt_options = Prompts().GetPromptOptions(prompt_idx)
        self.prompt_columns = Prompts().GetPromptColumns(prompt_idx)
        self.valid_columns = [col for col in self.prompt_columns if col is not None]
        # If no output path is provided, overwrite the original file.
        self.output_path = output_path if output_path else file_path

    def add_columns(self):
        # Read the CSV into a DataFrame.
        df = pd.read_csv(self.file_path)
        
        # Check if the 'response' column exists.
        if 'response' in df.columns:
            for col in self.prompt_columns:
                # Add the column if it does not exist.
                if col is not None and col not in df.columns:
                    df[col] = "ERROR"
            
            # Save the updated DataFrame to the output file.
            df.to_csv(self.output_path, index=False)
            print(f"Modified CSV saved to {self.output_path}")
        else:
            print("Column 'response' not found in the CSV. No modifications made.")
        
    
    def extract_answers(self, model_output, is_recursive=False):
        result = []

        # Replace literal "\n" with an actual newline character
        model_output = model_output.replace("\\n", "\n")

        # Normalize model output to lowercase for easier matching
        model_output = model_output.lower()

        # Preprocess options: remove leading/trailing whitespace and replace spaces with hyphens
        modified_options = self.prompt_options.copy()
        for i, options in enumerate(self.prompt_options):
            if options is None:
                modified_options[i] = None
            else:
                modified_options[i] = [option.strip().replace(" ", "-") for option in options]

        # Modify model_output to replace matching multi-word options with hyphenated versions
        for options in self.prompt_options:
            if options is not None:
                for option in options:
                    mod_option = option.strip().lower()
                    if " " in mod_option:
                        model_output = model_output.replace(mod_option, mod_option.replace(" ", "-"))

        # Construct lowercase categories and associated keys for extraction
        categories = []
        keys = []
        for key, opts in zip(self.prompt_columns, modified_options):
            if opts is not None and key is not None:
                lower_opts = {opt.lower() for opt in opts}
                categories.append(lower_opts)
                keys.append(key)

        num_cats = len(categories)
        if num_cats == 0:
            return []
        
        # Handle special case: if the substring "assistant" is found in model_output
        assistant_match = re.search(r'assistant', model_output, re.IGNORECASE)
        if assistant_match:
            # Extract everything after the matched "assistant"
            assistant_text = model_output[assistant_match.end():]
            
            # Recursively call extract_answers on the extracted text
            assistant_result = self.extract_answers(assistant_text, True)
            
            # If not all returned items are "error", then use the assistant_result
            if not all(item.lower() == "error" for item in assistant_result):
                return assistant_result

        # Handle special case: if possible answers are in parentheses
        paren_matches = re.findall(r'\(.*?([^)]+)\)', model_output, re.IGNORECASE)
        if len(paren_matches) == num_cats:
            # Recursively call extract_answers on the extracted text
            paren_result = self.extract_answers(" ".join(paren_matches), True)
            if all(item.lower() == "error" for item in paren_result):
                # If parentheses don't help, remove them from the string
                model_output = re.sub(r'\([^)]*\)', '', model_output)
            else:
                return paren_result

        # Remove all punctuation except for spaces and hyphens
        model_output = re.sub(r"[^\w\s\-]", "", model_output)
        tokens = model_output.split()
        
        # Determine max token length for each category to guide the backtracking
        max_tokens = []
        for opts in categories:
            max_len = max(len(opt.split()) for opt in opts)
            max_tokens.append(max_len)

        solutions = []

        # Recursive backtracking to find all valid combinations of category matches
        def backtrack(cat_idx, token_idx, current):
            if cat_idx == num_cats:
                solutions.append(current[:])
                return
            
            if token_idx >= len(tokens):
                return
            
            # Try skipping this token
            backtrack(cat_idx, token_idx + 1, current)

            # Try matching substrings of increasing length
            for l in range(max_tokens[cat_idx], 0, -1):
                if token_idx + l > len(tokens):
                    continue
                candidate = " ".join(tokens[token_idx: token_idx + l])
                if candidate in categories[cat_idx]:
                    backtrack(cat_idx + 1, token_idx + l, current + [candidate])

        # Start backtracking from the beginning
        backtrack(0, 0, [])

        # If no valid solution was found
        if not solutions:
            result = ["ERROR"] * num_cats
        else:
            # Choose the longest unambiguous answer for each category
            for cat_idx in range(num_cats):
                candidate_set = {sol[cat_idx] for sol in solutions}
                max_token_count = max(len(candidate.split()) for candidate in candidate_set)
                candidates_with_max = [c for c in candidate_set if len(c.split()) == max_token_count]
                if len(candidates_with_max) == 1:
                    result.append(candidates_with_max[0])
                else:
                    result.append("ERROR")
        
        # Replace hyphenated or modified answers with their original formatting
        for i in range(len(result)):
            for j in range(len(modified_options)):
                if modified_options[j] is not None:
                    for k in range(len(modified_options[j])):
                        if result[i].strip().lower() == modified_options[j][k].strip().lower() and result[i] != self.prompt_options[j][k].strip().lower():
                            result[i] = self.prompt_options[j][k].strip().lower()
        
        # Print debug message if there's an error and this is the top-level call
        if not is_recursive:
            has_error = False
            for r in result:
                if r.strip().lower() == "error":
                    has_error = True
                    break
            if has_error:
                print(f"Model output has an ERROR: {model_output}")

        return result

    def process_csv(self):
        print(f"Parsing: {self.file_path}")
        df = pd.read_csv(self.output_path)

        def update_row(row):
            # Call the extract_answers method on the 'response' column.
            answers = self.extract_answers(row['response'])
            
            # Update each corresponding column with the respective answer.
            for col, answer in zip(self.valid_columns, answers):
                row[col] = answer
            return row

        # Apply the update_row function to each row in the DataFrame.
        df = df.apply(update_row, axis=1)
        
        # Write the updated DataFrame back to the CSV file.
        df.to_csv(self.output_path, index=False)

    def run(self):
        self.add_columns()
        self.process_csv()

def main():
    id = 0
    if sys.argv[1] == "-d":
        id = 1
    
    file_path = sys.argv[id + 1]
    prompt_idx = int(sys.argv[id + 2])

    output_path = None

    if len(sys.argv) > id + 3:
        output_path = sys.argv[id + 3]

    if id == 0:
        csv_modifier = CSVModifier(file_path, prompt_idx, output_path)
        csv_modifier.run()
    else:
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.lower().endswith(".csv"):
                    csv_full_path = os.path.join(root, file)
                    csv_modifier = CSVModifier(csv_full_path, prompt_idx, output_path)
                    csv_modifier.run()


    # test_strings = [
    #     "1. yes 2. visible 3. male 4. child 5. White",
    #     "1. Number of People: yes (Options: yes / no) 2. Face Visibility: visible (Options: visible / not visible) 3. Gender: male (Options: male / female / other) 4. Age Group: teenager (Options: child / teenager / adult / middle-aged / senior) 5. Race/Ethnicity: White (Options: White / Black / Indian / East Asian / Southeast Asian / Middle Eastern / Latino)",
    #     "1. yes / no 2. visible / not visible 3. male / female / other 4. adult / middle-aged / senior 5. White / Southeast Asian / Middle Eastern",
    #     "no",
    #     "1. Number of People: yes / no (Answer: yes) 2. Face Visibility: visible / not visible (Answer: visible) 3. Gender: female / other (Answer: female) 4. Age Group: adult / middle-aged / senior (Answer: middle-aged) 5. Race/Ethnicity: East Asian / Southeast Asian (Answer: Southeast Asian)",
    #     "1. yes 2. not visible 3. female 4. adult 5. White",
    #     "1. yes 2. not visible 3. female 4. adult 5. Southeast asian",
    #     "1. yes 2. not visible visible 3. female 4. adult 5. Southeast asian",
    #     "1. Number of People: yes (1) 2. Face Visibility: visible (1) 3. Gender: female (1) 4. Age Group: adult (1) 5. Race/Ethnicity: Indian (1)",
    #     "1. Number of People: yes (1/1) 2. Face Visibility: visible (1/1) 3. Gender: male (1/1) 4. Age Group: adult (1/1) 5. Race/Ethnicity: White (1/1)",
    #     "['user \nPlease answer the following three multiple choice questions based only on the image. Be objective and make your choice without making unsupported assumptions. Provide each answer on a new line with the selected option. Do not restate the questions or include any extra text. 1. What is the gender of the person: male, female, or other. 2. Which age group does the person appear to belong to: child, teenager, adult, middle-aged, or senior. 3. What is the apparent race or ethnicity of the person: White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino.assistant\n1. male\n2. adult\n3. White']",
    #     "['USER: \nPlease answer the following three multiple choice questions based only on the image. Be objective and make your choice without making unsupported assumptions. Provide each answer on a new line with the selected option. Do not restate the questions or include any extra text. 1. What is the gender of the person: male, female, or other. 2. Which age group does the person appear to belong to: child, teenager, adult, middle-aged, or senior. 3. What is the apparent race or ethnicity of the person: White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino. ASSISTANT: 1. Male\n2. Adult\n3. Middle-aged']",
    #     "['USER: \nPlease answer the following three multiple choice questions based only on the image. Be objective and make your choice without making unsupported assumptions. Provide each answer on a new line with the selected option. Do not restate the questions or include any extra text. 1. What is the gender of the person: male, female, or other. 2. Which age group does the person appear to belong to: child, teenager, adult, middle-aged, or senior. 3. What is the apparent race or ethnicity of the person: White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino. ASSISTANT: 1. Male\n2. Adult\n3. Middle-Eastern']",
    #     "female child adult",
    # ]
    # for s in test_strings:
    #     answers = csv_modifier.extract_answers(s)
    #     print(f"String: {s}\n\t{answers}")

if __name__ == "__main__":
    main()

    
