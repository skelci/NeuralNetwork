from collections import Counter



class Tokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.current_id = 0


    def add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.current_id
            self.id_to_token[self.current_id] = token
            self.current_id += 1
            print(f"Token added: '{token}'")
            if self.current_id >= self.vocab_size:
                return False
        else:
            print(f"Token '{token}' already exists in the vocabulary.")
        return True
    

    def build_base_vocab(self, text):
        tokens = set()
        for char in text:
            if char not in tokens:
                tokens.add(char)
                self.add_token(char)


    def generate_new_token(self, text):
        tokens = self.tokenize(text)
        token_counts = Counter()
        for i in range(len(tokens)-1):
            token_pair = tokens[i] + tokens[i+1]
            token_counts[token_pair] += 1
        
        new_token = token_counts.most_common(1)[0][0]
        return self.add_token(new_token)


    def tokenize(self, text):
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            found_match = False
            # Iterate from the longest possible token length at the current position `i` down to 1.
            # The longest possible token starting at `i` would be text[i:n].
            # So, check lengths from (n-i) down to 1.
            for length in range(min(n - i, 16), 0, -1): 
                substring = text[i : i + length]
                if substring in self.token_to_id:
                    tokens.append(substring)
                    i += length  # Advance pointer by the length of the found token
                    found_match = True
                    break  # Found the longest match for this position, move to the next segment
            
            if not found_match:
                # This block is reached if no token in the vocabulary (not even a single character)
                # matches any substring starting at text[i]. This implies text[i] is an unknown character.
                # (e.g. if tokenizing text with characters not seen during build_base_vocab).
                # To prevent an infinite loop and handle such unknown characters:
                # add the single character text[i] as a token and advance the pointer.
                # This ensures progress through the text.
                # If build_base_vocab was called on this text, all its individual characters
                # are in token_to_id, so `length=1` would find a match, and this block
                # would not be hit. This makes it robust for general use.
                tokens.append(text[i])
                i += 1
        return tokens
    

    def build_vocab(self, text):
        self.build_base_vocab(text)
        while True:
            if not self.generate_new_token(text):
                break

        
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for token, token_id in self.token_to_id.items():
                # Escape special characters:
                # 1. Replace literal backslashes with '\\\\'
                # 2. Replace newlines with '\\n'
                # 3. Replace tabs with '\\t'
                # The order is important, especially escaping backslashes first.
                escaped_token = token.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
                f.write(f"{token_id}\t{escaped_token}\n")


    def load(self, path):
        # Reset current vocabulary and ID counter before loading
        self.token_to_id = {}
        self.id_to_token = {}
        self.current_id = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # Remove only the trailing newline character, preserving other whitespace
                line = line.rstrip('\n') 
                try:
                    # Split only on the first tab to separate ID from token
                    token_id_str, escaped_token = line.split("\t", 1)
                    token_id = int(token_id_str)
                except ValueError:
                    # Handles lines not in "int\tescaped_token" format
                    # (e.g., empty lines, lines without a tab, or non-integer token_id)
                    print(f"Warning: Skipping malformed line: '{line}'")
                    continue
                
                # Unescape special characters:
                # 1. Replace '\\t' with tab
                # 2. Replace '\\n' with newline
                # 3. Replace '\\\\' with backslash
                # The order is important to correctly reverse the escaping.
                token = escaped_token.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")
                
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                # Ensure current_id is set to the next available ID
                self.current_id = max(self.current_id, token_id + 1)




            
