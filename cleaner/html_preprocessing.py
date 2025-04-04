import re
from bs4 import BeautifulSoup
from cleaner.utils_html import strip_leading_symbols, fix_merged_phrases, plan_text_chunks

def clean_html_format(html:str)->list:
    # If there are no HTML tags, treat the text as plain text
    if '<' not in html:
        return plan_text_chunks(html)

    # Replace <br> and <br/> tags with newlines \n
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract text from HTML tags
    lines = []
    # Keep track of seen lines to avoid duplicates
    seen = set()

    # Handles cases where the text is not wrapped in any tag (e.g. beginning of the document)
    if soup.contents and isinstance(soup.contents[0], str):
        initial_text = soup.contents[0].strip()
        if initial_text:
            # Fix merged words and unsupported characters
            initial_text = fix_merged_phrases(initial_text)
            for line in initial_text.split('\n'):
                if line not in seen:
                    lines.append(line)
                    seen.add(line)
    # Extract text from HTML tags
    for tag in soup.find_all(['p', 'li', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = tag.get_text(separator='\n').strip()
        # Skip empty lines
        if not text:
            continue
        # Fix merged words and unsupported characters
        text = fix_merged_phrases(text)
        for line in text.split('\n'):
            if line not in seen:
                lines.append(line)
                seen.add(line)
                
    # In case no lines were extracted, extract lines from the entire text
    if not lines:
        raw = fix_merged_phrases(soup.get_text(separator='\n').strip())
        lines = raw.split('\n')

    # Extract text chunks
    text_chunks = []
    for line in lines:
        line = line.strip()
        # Skip empty lines or lines containing JavaScript junk code
        if not line or any(k in line for k in ["google.load", "function(", "$", "innerHTML"]):
            continue
        # Strip leading symbols like - or * from the beginning of the text and add to text_chunks
        text_chunks.append(strip_leading_symbols(line))

    return text_chunks

