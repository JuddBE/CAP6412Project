import numbers


class Prompts:
    prompts = [
        [  # Prompt 0
            "I am gonna ask 3 multiple choice questions please answer each to the best of your ability. Each of your three answers should be exactly one word!",
            "1. What gender of the person between male, female, other, or unknown?",
            "2. What age of the person child, teenager, adult, middle-aged, senior, or unknown?",
            "3. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, latino, or unknown?",
        ],
        [  # Prompt 1
            "I am gonna ask 3 multiple choice questions please answer each to the best of your ability. Each of your three answers should be exactly one word!",
            "1. What gender of the person between male, female, or other?",
            "2. What age of the person child, teenager, adult, middle-aged, or senior?",
            "3. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, or latino?",
        ],
        [  # Prompt 2
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
            ),
        ],
        [  # Prompt 3
            "Please answer the following five multiple choice questions based on the visual content of the image. Be objective and make your choice without making unsupported assumptions. Provide each answer on its own new line, with the selected choice!",
            "1. Is there only one person in the image yes or no?",
            "2. The persons face is visible or not-visible?",
            "3. What gender of the person between male, female, or other?",
            "4. What age of the person child, teenager, adult, middle-aged, or senior?",
            "5. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, or latino?",
        ],
        [  # Prompt 4
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
            ),
        ],
        [  # Prompt 5
            (
                "Please answer the following five multiple choice questions based only on the image. "
                "Be objective and make your choice without making unsupported assumptions. "
                "Provide each answer on a new line with the selected option. "
                "Do not restate the questions or include any extra text."
            ),
            (
                "1. Is there exactly one person in the image:"
                " yes or no."  # Putting this comment here so that the autoformatter keeps this on a separate line
            ),
            (
                "2. Is the person's face clearly visible:"
                " visible or not-visible."  # Putting this comment here so that the autoformatter keeps this on a separate line
            ),
            (
                "3. What is the gender of the person:"
                " male, female, or other."  # Putting this comment here so that the autoformatter keeps this on a separate line
            ),
            (
                "4. Which age group does the person appear to belong to:"
                " child, teenager, adult, middle-aged, or senior."  # Putting this comment here so that the autoformatter keeps this on a separate line
            ),
            (
                "5. What is the apparent race or ethnicity of the person:"
                " White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino."  # Putting this comment here so that the autoformatter keeps this on a separate line
            ),
        ],
        [  # Prompt 6
            (
                "Please answer the following three multiple choice questions based only on the image. "
                "Be objective and make your choice without making unsupported assumptions. "
                "Provide each answer on a new line with the selected option. "
                "Do not restate the questions or include any extra text."
            ),
            ("1. What is the gender of the person:" " male, female, or other."),
            (
                "2. Which age group does the person appear to belong to:"
                " child, teenager, adult, middle-aged, or senior."
            ),
            (
                "3. What is the apparent race or ethnicity of the person:"
                " White, Black, Indian, East-Asian, Southeast-Asian, Middle-Eastern, or Latino."
            ),
        ],
    ]

    # Use None if there is not an associated column
    prompt_columns = [
        [  # Prompt 0
            None,  # question
            "gender",  # answer 1
            "age",  # answer 2
            "race",  # answer 3
        ],
        [  # Prompt 1
            None,  # question
            "gender",  # answer 1
            "age",  # answer 2
            "race",  # answer 3
        ],
        [  # Prompt 2
            None,  # question
            "one_person",  # answer 1
            "face_visible",  # answer 2
            "gender",  # answer 3
            "age",  # answer 4
            "race",  # answer 5
        ],
        [  # Prompt 3
            None,  # question
            "one_person",  # answer 1
            "face_visible",  # answer 2
            "gender",  # answer 3
            "age",  # answer 4
            "race",  # answer 5
        ],
        [  # Prompt 4
            None,  # question
            "one_person",  # answer 1
            "face_visible",  # answer 2
            "gender",  # answer 3
            "age",  # answer 4
            "race",  # answer 5
        ],
        [  # Prompt 5
            None,  # question
            "one_person",  # answer 1
            "face_visible",  # answer 2
            "gender",  # answer 3
            "age",  # answer 4
            "race",  # answer 5
        ],
        [  # Prompt 6
            None,  # question
            "gender",  # answer 1
            "age",  # answer 2
            "race",  # answer 3
        ],
    ]

    # Use None if there is unlimited options or the answer doesn't matter
    prompt_options = [
        [  # Prompt 0
            None,
            ["male", "female", "other", "unknown"],
            ["child", "teenager", "adult", "middle-aged", "senior", "unknown"],
            [
                "white",
                "black",
                "indian",
                "east-asian",
                "southeast-asian",
                "middle-eastern",
                "latino",
                "unknown",
            ],
        ],
        [  # Prompt 1
            None,
            ["male", "female", "other"],
            ["child", "teenager", "adult", "middle-aged", "senior"],
            [
                "white",
                "black",
                "indian",
                "east-asian",
                "southeast-asian",
                "middle-eastern",
                "latino",
            ],
        ],
        [  # Prompt 2
            None,
            ["yes", "no"],
            ["visible", "not visible"],
            ["male", "female", "other"],
            ["child", "teenager", "adult", "middle-aged", "senior"],
            [
                "white",
                "black",
                "indian",
                "east asian",
                "southeast asian",
                "middle eastern",
                "latino",
            ],
        ],
        [  # Prompt 3
            None,
            ["yes", "no"],
            ["visible", "not-visible"],
            ["male", "female", "other"],
            ["child", "teenager", "adult", "middle-aged", "senior"],
            [
                "white",
                "black",
                "indian",
                "east-asian",
                "southeast-asian",
                "middle-eastern",
                "latino",
            ],
        ],
        [  # Prompt 4
            None,
            ["yes", "no"],
            ["visible", "not visible", "yes", "no"],
            ["male", "female", "other"],
            ["child", "teenager", "adult", "middle-aged", "senior"],
            [
                "white",
                "black",
                "indian",
                "east asian",
                "southeast asian",
                "middle eastern",
                "latino",
            ],
        ],
        [  # Prompt 5
            None,
            ["yes", "no"],
            ["visible", "not-visible", "yes", "no"],
            ["male", "female", "other"],
            ["child", "teenager", "adult", "middle-aged", "senior"],
            [
                "white",
                "black",
                "indian",
                "east-asian",
                "southeast-asian",
                "middle-eastern",
                "latino",
            ],
        ],
        [  # Prompt 6
            None,
            ["male", "female", "other"],
            ["child", "teenager", "adult", "middle-aged", "senior"],
            [
                "white",
                "black",
                "indian",
                "east-asian",
                "southeast-asian",
                "middle-eastern",
                "latino",
            ],
        ],
    ]

    def GetPrompt(prompt):
        if isinstance(prompt, numbers.Number):
            return " ".join(Prompts.prompts[prompt]).strip()
        elif isinstance(prompt, list):
            return " ".join(prompt).strip()

    def GetPromptList(index):
        return Prompts.prompts[index]

    def GetPromptOptions(index):
        return Prompts.prompt_options[index]

    def GetPromptColumns(index):
        return Prompts.prompt_columns[index]
