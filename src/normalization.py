import unicodedata
import re
import math

def normalize_text(text):
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)
    # Unicode normalize
    # NFKC first folds compatibility characters (full-width forms, ligatures),
    # then NFKD lets us remove accents while retaining non-Latin letters.
    text = unicodedata.normalize('NFKC', text)
    text = unicodedata.normalize('NFKD', text)
    # Strip Latin accents (cafe -> cafe), but keep vowel signs and other
    # combining marks in scripts where they are part of the written letter.
    kept = []
    last_base_name = ""
    for char in text:
        if unicodedata.combining(char):
            if "LATIN" not in last_base_name:
                kept.append(char)
            continue
        kept.append(char)
        last_base_name = unicodedata.name(char, "")
    text = "".join(kept)
    text = text.replace('&', ' and ').replace('ß', 'ss').replace('ẞ', 'SS')
    # Lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = "".join(
        char if char.isalnum() or char.isspace()
        or unicodedata.category(char).startswith('M') else ' '
        for char in text
    )
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<!\w)0+(\d+)(?!\w)', r'\1', text)
    # Strip
    return text.strip()

def normalize_business_name(name):
    norm = normalize_text(name)
    # legal suffixes mapping
    replacements = {
        r'\bcorporation\b': 'corp', r'\bcompany\b': 'co',
        r'\bprivate\b': 'pvt', r'\blimited\b': 'ltd',
        r'\bincorporated\b': 'inc', r'\b sociedades? anonimas?\b': 'sa',
        r'\bgesellschaft mit beschrankter haftung\b': 'gmbh',
        r'\bsociete a responsabilite limitee\b': 'sarl',
        r'\bpublic limited company\b': 'plc',
    }
    for k, v in replacements.items():
        norm = re.sub(k, v, norm)
    # clean extra space just in case
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm

def normalize_address(address):
    norm = normalize_text(address)
    replacements = {
        r'\broad\b': 'rd',
        r'\bstreet\b': 'st',
        r'\bavenue\b': 'ave',
        r'\bboulevard\b': 'blvd',
        r'\blane\b': 'ln',
        r'\bsector\b': 'sec'
        , r'\bstrasse\b': 'str', r'\bstreet\b': 'st',
        r'\broute\b': 'rte', r'\bchemin\b': 'chem',
        r'\bnumero\b': 'no'
    }
    for k, v in replacements.items():
        norm = re.sub(k, v, norm)
    # clean extra space
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm

def generate_phonetic_key(name):
    """
    Soundex algorithm implementation:
    1. Retain first letter.
    2. Drop all occurrences of a, e, i, o, u, y, h, w.
    3. Replace consonants with digits as follows:
       b, f, p, v -> 1
       c, g, j, k, q, s, x, z -> 2
       d, t -> 3
       l -> 4
       m, n -> 5
       r -> 6
    4. Two adjacent letters with the same number are coded as a single number.
       Letters with the same number separated by 'h' or 'w' are coded as a single number.
    5. Pad with zeroes or truncate to ensure 4 characters.
    """
    norm = normalize_text(name)
    # remove spaces
    norm = re.sub(r'\s+', '', norm)
    if not norm:
        return ""
    # remove non-alpha for soundex
    norm = re.sub(r'[^a-z]', '', norm)
    if not norm:
        return ""
    first_char = norm[0].upper()
    
    mapping = {
        'b':'1', 'f':'1', 'p':'1', 'v':'1',
        'c':'2', 'g':'2', 'j':'2', 'k':'2', 'q':'2', 's':'2', 'x':'2', 'z':'2',
        'd':'3', 't':'3',
        'l':'4',
        'm':'5', 'n':'5',
        'r':'6'
    }
    
    encoded = []
    last_digit = mapping.get(norm[0], '')
    for char in norm[1:]:
        if char in ('h', 'w'):
            continue
        digit = mapping.get(char, '')
        if digit:
            if digit != last_digit:
                encoded.append(digit)
                last_digit = digit
        else:
            # vowel
            last_digit = ''
    
    soundex = (first_char + "".join(encoded)).ljust(4, '0')[:4]
    return soundex

def generate_sorted_token_key(name):
    norm = normalize_business_name(name)
    tokens = sorted(norm.split())
    return " ".join(tokens)

