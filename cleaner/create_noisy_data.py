import random 

def strip_trailing_comma(word:str)->str:
    '''Helper for word merge 
    Removes trailing comma from a word if it exists'''
    return word[:-1] if word.endswith(',') else word

def is_merged_allowed(word1:str, word2:str)->bool:
    '''Checks if two words can be merged
    Protected words: email, urls, numbers
    '''
    merged = word1 + word2
    if any(char.isdigit() for char in merged):
        return False
    if '@' in merged or 'http' in merged or 'www' in merged:
        return False
    return True

def add_junk_html(word:str)->str:
    '''Adds junk HTML tokens to the text chunk
    with 50% probability at the beginning and 50% at the end
    '''
    junk_tokens = ['<br>', '<br/>', '&nbsp;', '<div>', '<span>', '</p>', '<p>', "$", "google.load", "innerHTML"]
    if random.random() < 0.5:
        return word + random.choice(junk_tokens)
    else:
        return random.choice(junk_tokens) + word

def add_noise(text_chunk:list, merge_prob:float =0.2, junk_prob:float=0.05)->str:
    '''Adds noise to the text chunk'''
    words = text_chunk.split()
    new_noisy_words = []  
    i = 0
    while i < len(words):
        # Merge words if not the last word and merge_prob is met
        if i < len(words) - 1 and random.random() < merge_prob:
            w1, w2 = words[i], words[i+1]
            # Check if words can be merged
            if is_merged_allowed(w1, w2):
                new_noisy_words.append(strip_trailing_comma(w1) + strip_trailing_comma(w2))
                # Counter for the next word
                i += 2
                continue
        # Add junk HTML token with junk_prob or keep the word
        if random.random() < junk_prob:
            new_noisy_words.append(add_junk_html(words[i]))
        else:
            new_noisy_words.append(words[i])
        i += 1
    return ' '.join(new_noisy_words)