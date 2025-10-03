import re

# reflection map to convert user input to Eliza-style responses
reflections = {
    "am": "are",
    "was": "were",
    "i": "you",
    "i'd": "you would",
    "i'll": "you will",
    "my": "your",
    "are": "am",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "I",
    "me": "you"
}
# define substitution rules as a list of tuples (pattern, response)
substitution_rules = [
    (r'\bI AM (depressed|sad|unhappy)\b', r'I AM SORRY TO HEAR YOU ARE \1.'),
    (r'\bI FEEL (depressed|sad|unhappy)\b', r'I AM SORRY TO HEAR YOU FEEL \1.'),
    (r"\bI'M (depressed|sad|unhappy)\b", r'I AM SORRY TO HEAR YOU ARE \1.'),
    (r'\bMY (.*)', r'TELL ME MORE ABOUT YOUR \1.'),
    (r'\bI NEED (.*)', r'WHAT WOULD IT MEAN TO YOU IF YOU GOT \1?'),
    (r"\bWHY DON'T YOU ([^\?]*)\??", r"DO YOU REALLY THINK I DON'T \1?"),
    (r"\bWHY CAN'T I ([^\?]*)\??", r'DO YOU THINK YOU SHOULD BE ABLE TO \1?'),
    (r"\bI CAN'T (.*)", r'HOW DO YOU KNOW YOU CAN\'T \1?'),
    (r'\bI AM (.*)', r'HOW LONG HAVE YOU BEEN \1?'),
    (r"\bI'M (.*)", r'HOW DOES BEING \1 MAKE YOU FEEL?'),
    (r'\bI THINK (.*)', r'DO YOU DOUBT \1?'),
    (r'\bI (.*) YOU\b', r'WHY DO YOU \1 ME?'),
    (r'\bYOU ARE (.*)', r'WHAT MAKES YOU THINK I AM \1?'),
    (r'\bYOU (.*)', r'WHY DO YOU SAY I \1?'),
    (r'.*\bALWAYS\b.*', r'CAN YOU THINK OF A SPECIFIC EXAMPLE?'),
    (r'.*\bALL\b.*', r'IN WHAT WAY?'),
    (r'.*', r'PLEASE TELL ME MORE.')
]

# reflect pronouns in captured input
def reflect(fragment):
    tokens = fragment.lower().split()
    reflected = [reflections.get(word, word) for word in tokens]
    return ' '.join(reflected)

# function to get Eliza-like responses based on user input
def eliza_response(user_input):
    # lop through the list of substitution rules
    for pattern, response in substitution_rules:
        match = re.search(pattern, user_input, re.IGNORECASE)
        # check if the current pattern matches the user input
        if match:
            # reflect the captured group(s) before inserting into the response
            groups = match.groups()
            reflected = [reflect(g) for g in groups]
            
            # replace backreferences like \1 with reflected context
            for i, val in enumerate(reflected, start=1):
                response = response.replace(f"\\{i}", val.upper())
            return response
        
    # default response if no pattern matches
    return "PLEASE TELL ME MORE."

# main loop to interact with user
def eliza_chat():
    print("Hello, I am Eliza. How can I help you today?")
    
    while True:
        user_input = input("You: ")
        
        # end conversation if user says "bye"
        if user_input.lower() == 'bye':
            print("Eliza: Goodbye! Take care.")
            break 
        
        # generate Eliza's response
        response = eliza_response(user_input)
        print(f"Eliza: {response}")

# test cases
def run_tests():
    test_cases = [
        "I'm sad",                    # Reflected: YOU ARE SAD
        "I am unhappy",               # Reflected: YOU ARE UNHAPPY
        "My mother takes care of me", # Reflected: YOUR MOTHER TAKES CARE OF YOU
        "I need help",                # Reflected: YOU NEED HELP
        "You are not very aggressive",# Reflected: I AM NOT VERY AGGRESSIVE
        "You don't argue with me",    # Reflected: I DON'T ARGUE WITH YOU
        "Men are all alike",          # Trigger: ALL
        "They're always bugging us",  # Trigger: ALWAYS
        "I love you"                  # Reflected: I LOVE ME → WHY DO YOU SAY I LOVE ME?
    ]

    print("Running test cases...\n")
    for i, input_text in enumerate(test_cases, 1):
        response = eliza_response(input_text)
        print(f"Test {i}: {input_text}")
        print(f"Response: {response}\n")

# start Eliza chatbot
if __name__ == "__main__":
    eliza_chat()
    #run_tests()