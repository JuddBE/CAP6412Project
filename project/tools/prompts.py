class Prompts:
    def __init__(self):
        self.prompts = [
            ( # Prompt 0
                "I am gonna ask 3 multiple choice questions please answer each to the best of your ability. Each of your three answers should be exactly one word!"
                " 1. What gender of the person between male, female, other, or unknown?"
                " 2. What age of the person child, teenager, adult, middle-aged, senior, or unknown?"
                " 3. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, latino, or unknown?"
            ),
            ( # Prompt 1
                "I am gonna ask 3 multiple choice questions please answer each to the best of your ability. Each of your three answers should be exactly one word!"
                " 1. What gender of the person between male, female, or other?"
                " 2. What age of the person child, teenager, adult, middle-aged, or senior?"
                " 3. What race of the person between white, black, indian, east-asian, southeast-asian, middle-eastern, or latino?"
            ),
        ]

    def GetPrompt(self, index):
        return self.prompts[index]