import re

def strip_leading_symbols(text:str)->str:
    '''Strips leading symbols like - or * from the beginning of the text'''
    # Remove leading dash or asterisk e.g. "- word" -> "word" or "* word" -> "word"
    return re.sub(r'^[-*]\s*', '', text)

def fix_merged_phrases(text:str)->str:
    '''Fixes merged phrases in text and removes unsupported characters'''
    #Replace problematic characters with space or dash
    text = text.replace('\xa0', ' ')# non-breaking space
    text = text.replace('\t', ' ')  # literal escaped tab
    text = text.replace('\n', ' ')  # literal newline
    text = text.replace('	', ' ') # actual tab
    text = text.replace('•', '-')   # bullet (•)
    text = text.replace('‣', '-')   # triangular bullet (‣)
    text = text.replace('∙', '-')   # dot operator (∙)
    text = text.replace('–', '-')   # en-dash
    text = text.replace('—', '-')   # em-dash

    # Remove control characters and other non-printable characters
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Add newline before dash bullets stuck to previous word (eg. "word- word" -> "word\n- word")
    text = re.sub(r'(?<=[a-zäöüß])-(?=\s*[A-ZÄÖÜ])', r'\n- ', text)

    # Add space after lowercase-uppercase merges (eg. "wordWord" -> "word Word")
    text = re.sub(r'(?<=[a-zäöüß])(?=[A-ZÄÖÜ])', r' ', text)
    
    # Add space after asterisk if missing (eg. "word*word" -> "word *word")
    text = re.sub(r'(\w+)\*(?![\s\n])(\w+)', r'\1 \2', text)
    return text

def plan_text_chunks(text:str)->list:
    '''Extracts text chunks from plain text that are separated by newlines'''
    chunks = [
        strip_leading_symbols(line.strip())
        for line in text.split('\n')
        if line.strip()
    ]
    return chunks