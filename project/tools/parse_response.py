import sys
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
        
    
    def extract_answers(self, model_output):
        result = []

        model_output = model_output.lower()

        modified_options = self.prompt_options.copy()
        for i, options in enumerate(self.prompt_options):
            if options is None:
                modified_options[i] = None
            else:
                modified_options[i] = [option.strip().replace(" ", "-") for option in options]

        for options in self.prompt_options:
            if options is not None:
                for option in options:
                    mod_option = option.strip().lower()
                    if " " in mod_option:
                        model_output = model_output.replace(mod_option, mod_option.replace(" ", "-"))

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

        # Remove any parenthetical content which may contain options
        model_output = re.sub(r"\(option[^)]*\)", "", model_output).strip()

        # Check for answers that might exist in parantheses
        paren_matches = re.findall(r'\(.*?([^)]+)\)', model_output, re.IGNORECASE)
        if len(paren_matches) == num_cats:
            paren_result = self.extract_answers(" ".join(paren_matches))
            return paren_result

        # Remove punctuation characters except spaces and hyphens (which may appear within words)
        model_output = re.sub(r"[^\w\s\-]", "", model_output)

        tokens = model_output.split()
        
        max_tokens = []
        for opts in categories:
            max_len = max(len(opt.split()) for opt in opts)
            max_tokens.append(max_len)

        solutions = []

        def backtrack(cat_idx, token_idx, current):
            if cat_idx == num_cats:
                solutions.append(current[:])
                return
            
            if token_idx >= len(tokens):
                return
            
            backtrack(cat_idx, token_idx + 1, current)

            for l in range(max_tokens[cat_idx], 0, -1):
                if token_idx + l > len(tokens):
                    continue
                candidate = " ".join(tokens[token_idx: token_idx + l])
                if candidate in categories[cat_idx]:
                    backtrack(cat_idx + 1, token_idx + l, current + [candidate])

        backtrack(0, 0, [])

        # print(f"solutions: {solutions}")

        # def backtrack(cat_idx, token_idx, current):
        #     if cat_idx == num_cats:
        #         solutions.append(current[:])
        #         return
            
        #     for i in range(token_idx, len(tokens)):
        #         for l in range(1, max_tokens[cat_idx] + 1):
        #             if i + l > len(tokens):
        #                 break
        #             candidate = " ".join(tokens[i:i+l])

        #             if candidate in categories[cat_idx]:
        #                 backtrack(cat_idx + 1, i + l, current + [candidate])

        # for start in range(len(tokens)):
        #     backtrack(0, start, [])

        if not solutions:
            result = ["ERROR"] * num_cats
        else:
            for cat_idx in range(num_cats):
                candidate_set = {sol[cat_idx] for sol in solutions}
                max_token_count = max(len(candidate.split()) for candidate in candidate_set)
                candidates_with_max = [c for c in candidate_set if len(c.split()) == max_token_count]
                if len(candidates_with_max) == 1:
                    result.append(candidates_with_max[0])
                else:
                    result.append("ERROR")
        
        for i in range(len(result)):
            for j in range(len(modified_options)):
                if modified_options[j] is not None:
                    for k in range(len(modified_options[j])):
                        if result[i].strip().lower() == modified_options[j][k].strip().lower() and result[i] != self.prompt_options[j][k].strip().lower():
                            result[i] = self.prompt_options[j][k].strip().lower()

        return result

        # allowed_options = {
        #     'one_person': self.prompt_options[1],
        #     'face_visible': self.prompt_options[2],
        #     'gender': self.prompt_options[3],
        #     'age': self.prompt_options[4],
        #     'race': self.prompt_options[5],
        # }
        # keys = ['one_person', 'face_visible', 'gender', 'age', 'race']

        allowed_options = {
            key: self.prompt_options[i]
            for i, key in enumerate(self.prompt_columns)
            if key is not None and self.prompt_options[i] is not None
        }

        segments = re.findall(r"(\d+\.\s*.*?)(?=\d+\.\s*|$)", model_output, flags=re.DOTALL)

        if len(segments) == 0:
            segments = re.findall(r'\S+', model_output)
        if len(segments) < len(self.prompt_options) - 1:
            segments += [""] * (5 - len(segments))
        
        answers = []
        has_error = False

        for idx, key in enumerate(self.valid_columns):
            seg = segments[idx].strip().lower()

            # Remove the questions from seg
            for phrase in self.text_prompts:
                if phrase is not None:
                    phrase = phrase.lower()
                    if phrase in seg:
                        seg = re.sub(re.escape(phrase), "", seg, flags=re.IGNORECASE).strip()

            # Remove any parenthetical content which may contain options
            seg = re.sub(r"\(option[^)]*\)", "", seg).strip()

            # If a colon (:) exists, assume the answer is after it
            if ":" in seg:
                candidate = seg.split(":")[-1].strip()
            else:
                candidate = seg.strip()

            # print(f"A: {candidate}")

            # Get allowed options for the key
            options = allowed_options.get(key, [])

            # First, check for an exact match
            exact_matches = [option.lower() for option in options if candidate == option.lower()]
            if len(exact_matches) == 1:
                found_option = exact_matches[0]
                # print(f"B: {found_option}")
            else:
                # Otherwise, perform a word-boundary search with the allowed options.
                # Sort options in descending order by length to ensure longer phrases (like "not visible")
                # are considered before shorter ones.
                options_sorted = sorted(options, key=lambda x: len(x), reverse=True)
                matches = []
                for option in options_sorted:
                    option = option.lower()
                    # print(f"C: {option}")
                    pattern = rf'(?:(?<=^)|(?<=[\s\.,;:!?\-\(\)])){re.escape(option)}(?=$|[\s\.,;:!?\-\(\)])'
                    if re.search(pattern, candidate):
                        accounted_for = False
                        for m in matches:
                            if " " + option + " " in " " + m + " ":
                                accounted_for = True
                        if not accounted_for:
                            matches.append(option)
                            # print(f"D: {option}")
                if len(matches) == 1:
                    found_option = matches[0]
                    # print(f"E: {found_option}")
                else:
                    found_option = None

            # Sort options in descending order of length to check longer phrases first
            # options_sorted = sorted(options, key=lambda x: len(x), reverse=True)

            # # Find all options matching as separate words (use boundaries)
            # matches = []
            # for option in options_sorted:
            #     option = option.lower()
                
            #     # Pattern that only matches when the option appears as a distinct word.
            #     pattern = rf'(?:(?<=^)|(?<=[\s\.,;:!?\-])){re.escape(option)}(?=$|[\s\.,;:!?\-])'
            #     if re.search(pattern, candidate):
            #         matches.append(option)

            # # If candidate contains more than one valid option or none, record an error
            # if len(matches) != 1 or candidate == "":
            #     answers.append("ERROR")
            #     has_error = True
            # else:
            #     answers.append(matches[0])

            # # Compare the candidate against the allowed options for this key
            # found_option = None
            # for option in allowed_options.get(key, []):
            #     option = option.lower()
            #     if re.search(rf'(?:(?<=^)|(?<=[\s\.,;:!\?\-])){re.escape(option)}(?=$|[\s\.,;:!\?\-])', candidate.lower()):
            #         found_option = option
            #         break
            
            if found_option is None or candidate == "":
                answers.append("ERROR")
                has_error = True
            else:
                answers.append(found_option)

        if has_error:
            print(f"Model output has an ERROR: {model_output}")
        
        return answers

    def process_csv(self):
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
    file_path = sys.argv[1]
    prompt_idx = int(sys.argv[2])

    output_path = None

    if len(sys.argv) > 3:
        output_path = sys.argv[3]

    csv_modifier = CSVModifier(file_path, prompt_idx, output_path)
    csv_modifier.run()
    # test_strings = [
    #     "1. yes 2. visible 3. male 4. child 5. White",
    #     "1. Number of People: yes (Options: yes / no) 2. Face Visibility: visible (Options: visible / not visible) 3. Gender: male (Options: male / female / other) 4. Age Group: teenager (Options: child / teenager / adult / middle-aged / senior) 5. Race/Ethnicity: White (Options: White / Black / Indian / East Asian / Southeast Asian / Middle Eastern / Latino)",
    #     "1. yes / no 2. visible / not visible 3. male / female / other 4. adult / middle-aged / senior 5. White / Southeast Asian / Middle Eastern",
    #     "no",
    #     "1. Number of People: yes / no (Answer: yes) 2. Face Visibility: visible / not visible (Answer: visible) 3. Gender: female / other (Answer: female) 4. Age Group: adult / middle-aged / senior (Answer: middle-aged) 5. Race/Ethnicity: East Asian / Southeast Asian (Answer: Southeast Asian)",
    #     "1. yes 2. not visible 3. female 4. adult 5. White",
    #     "1. yes 2. not visible 3. female 4. adult 5. Southeast asian",
    #     "1. yes 2. not visible visible 3. female 4. adult 5. Southeast asian"
    # ]
    # for s in test_strings:
    #     answers = csv_modifier.extract_answers(s)
    #     print(f"String: {s}\n\t{answers}")

if __name__ == "__main__":
    main()

    
