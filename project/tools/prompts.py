import numbers

class Prompts:
    def __init__(self):
        self.prompts = [
            [ # Prompt 0
                "I am gonna ask 3 multiple choice questions please answer each to the best of your ability. Each of your three answers should be exactly one word!",
                "1. What gender of the person between male, female, other, or unknown?",
                "2. What age of the person child, teenager, adult, middle-aged, senior, or unknown?",
                "3. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, latino, or unknown?",
            ],
            [ # Prompt 1
                "I am gonna ask 3 multiple choice questions please answer each to the best of your ability. Each of your three answers should be exactly one word!",
                "1. What gender of the person between male, female, or other?",
                "2. What age of the person child, teenager, adult, middle-aged, or senior?",
                "3. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, or latino?",
            ],
            [ # Prompt 2
                (
                    "Analyze the image carefully and answer the following five questions solely based on what is seen. "
                    "Do not restate the questions or add any extra commentary. Your response must consist exclusively "
                    "of the chosen options for each question, with no additional text or context. "
                    "Format your answers exactly as follows (each answer on a new line): "
                    "1. answer 2. answer 3. answer 4. answer 5. answer. "
                    "Only output the option, nothing more."
                ),
                (
                    "1. Number of People:"
                    " Is there exactly one person in the image?"
                    " (Options: yes / no)"
                ),
                (
                    "2. Face Visibility:"
                    " Is the person's face clearly visible?"
                    " (Options: visible / not visible)"
                ),
                (
                    "3. Gender:"
                    " What is the gender of the person?"
                    " (Options: male / female / other)"
                ),
                (
                    "4. Age Group:"
                    " Which age group does the person appear to belong to?"
                    " (Options: child / teenager / adult / middle-aged / senior)"
                ),
                (
                    "5. Race/Ethnicity:"
                    " What is the apparent race or ethnicity of the person?"
                    " (Options: White / Black / Indian / East Asian / Southeast Asian / Middle Eastern / Latino)"
                )
            ],
            [ # Prompt 3
                "Please answer the following five multiple choice questions based on the visual content of the image. Be objective and make your choice without making unsupported assumptions. Provide each answer on its own new line, with the selected choice!",
                "1. Is there only one person in the image yes or no?",
                "2. The persons face is visible or not-visible?",
                "3. What gender of the person between male, female, or other?",
                "4. What age of the person child, teenager, adult, middle-aged, or senior?",
                "5. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, or latino?",
            ],
            [ # Prompt 4
                (
                    "Please answer the following five multiple choice questions based only on the image. "
                    "Be objective and make your choice without making unsupported assumptions. "
                    "Provide each answer on a new line with the selected option. "
                    "Do not restate the questions or include any extra text."
                ),
                (
                    "1. Number of People:"
                    " Is there exactly one person in the image?"
                    " (Options: yes / no)"
                ),
                (
                    "2. Face Visibility:"
                    " Is the person's face clearly visible?"
                    " (Options: visible / not visible)"
                ),
                (
                    "3. Gender:"
                    " What is the gender of the person?"
                    " (Options: male / female / other)"
                ),
                (
                    "4. Age Group:"
                    " Which age group does the person appear to belong to?"
                    " (Options: child / teenager / adult / middle-aged / senior)"
                ),
                (
                    "5. Race/Ethnicity:"
                    " What is the apparent race or ethnicity of the person?"
                    " (Options: White / Black / Indian / East Asian / Southeast Asian / Middle Eastern / Latino)"
                )
            ],
            [ # Prompt 5
                (
                    "Please answer the following five multiple choice questions based only on the image. "
                    "Be objective and make your choice without making unsupported assumptions. "
                    "Provide each answer on a new line with the selected option. "
                    "Do not restate the questions or include any extra text."
                ),
                (
                    "1. Is there exactly one person in the image:"
                    " yes or no."
                ),
                (
                    "2. Is the person's face clearly visible:"
                    " visible or not-visible."
                ),
                (
                    "3. What is the gender of the person:"
                    " male, female, or other."
                ),
                (
                    "4. Which age group does the person appear to belong to:"
                    " child, teenager, adult, middle-aged, or senior."
                ),
                (
                    "5. What is the apparent race or ethnicity of the person:"
                    " White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino."
                )
            ],
            [ # Prompt 6
                (
                    "Please answer the following three multiple choice questions based only on the image. "
                    "Be objective and make your choice without making unsupported assumptions. "
                    "Provide each answer on a new line with the selected option. "
                    "Do not restate the questions or include any extra text."
                ),
                (
                    "1. What is the gender of the person:"
                    " male, female, or other."
                ),
                (
                    "2. Which age group does the person appear to belong to:"
                    " child, teenager, adult, middle-aged, or senior."
                ),
                (
                    "3. What is the apparent race or ethnicity of the person:"
                    " White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino."
                )
            ],
        ]

        # Use None if there is not an associated column
        self.prompt_columns = [
            [ # Prompt 0
                None,
                "gender",
                "age",
                "race"
            ],
            [ # Prompt 1
                None,
                "gender",
                "age",
                "race"
            ],
            [ # Prompt 2
                None,
                "one_person",
                "face_visible",
                "gender",
                "age",
                "race"
            ],
            [ # Prompt 3
                None,
                "one_person",
                "face_visible",
                "gender",
                "age",
                "race"
            ],
            [ # Prompt 4
                None,
                "one_person",
                "face_visible",
                "gender",
                "age",
                "race"
            ],
            [ # Prompt 5
                None,
                "one_person",
                "face_visible",
                "gender",
                "age",
                "race"
            ],
            [ # Prompt 6
                None,
                "gender",
                "age",
                "race"
            ],
        ]

        # Use None if there is unlimited options or the answer doesn't matter
        self.prompt_options = [
            [ # Prompt 0
                None,
                ["male", "female", "other", "unknown"],
                ["child", "teenager", "adult", "middle-aged", "senior", "unknown"],
                ["white", "black", "indian", "east-asian", "southeast-asian", "middle-eastern", "latino", "unknown"],
            ],
            [ # Prompt 1
                None,
                ["male", "female", "other"],
                ["child", "teenager", "adult", "middle-aged", "senior"],
                ["white", "black", "indian", "east-asian", "southeast-asian", "middle-eastern", "latino"],
            ],
            [ # Prompt 2
                None,
                ["yes", "no"],
                ["visible" , "not visible"],
                ["male" , "female" , "other"],
                ["child" , "teenager" , "adult" , "middle-aged" , "senior"],
                ["white" , "black" , "indian" , "east asian" , "southeast asian" , "middle eastern" , "latino"]
            ],
            [ # Prompt 3
                None,
                ["yes", "no"],
                ["visible" , "not visible"],
                ["male" , "female" , "other"],
                ["child" , "teenager" , "adult" , "middle-aged" , "senior"],
                ["white" , "black" , "indian" , "east-asian" , "southeast-asian" , "middle-eastern" , "latino"]
            ],
            [ # Prompt 4
                None,
                ["yes", "no"],
                ["visible" , "not visible", "yes", "no"],
                ["male" , "female" , "other"],
                ["child" , "teenager" , "adult" , "middle-aged" , "senior"],
                ["white" , "black" , "indian" , "east asian" , "southeast asian" , "middle eastern" , "latino"]
            ],
            [ # Prompt 5
                None,
                ["yes", "no"],
                ["visible" , "not-visible", "yes", "no"],
                ["male" , "female" , "other"],
                ["child" , "teenager" , "adult" , "middle-aged" , "senior"],
                ["white" , "black" , "indian" , "east-asian" , "southeast-asian" , "middle-eastern" , "latino"]
            ],
            [ # Prompt 6
                None,
                ["male", "female", "other"],
                ["child", "teenager", "adult", "middle-aged", "senior"],
                ["white", "black", "indian", "east-asian", "southeast-asian", "middle-eastern", "latino"],
            ],
        ]

    def GetPrompt(self, prompt):
        if isinstance(prompt, numbers.Number):
            return " ".join(self.prompts[prompt]).strip()
        elif isinstance(prompt, list):
            return " ".join(prompt).strip()
    
    def GetPromptList(self, index):
        return self.prompts[index]
    
    def GetPromptOptions(self, index):
        return self.prompt_options[index]
    
    def GetPromptColumns(self, index):
        return self.prompt_columns[index]