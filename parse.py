import re
from pathlib import Path
text = Path('duck.html').read_bytes().decode('utf-8', errors='ignore')
pattern = re.compile(r'https://[^\'\" ]+', re.IGNORECASE)
links = sorted(set(pattern.findall(text)))
for link in links:
    print(link)
