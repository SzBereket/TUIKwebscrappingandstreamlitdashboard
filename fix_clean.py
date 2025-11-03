# -*- coding: utf-8 -*-
import re
from pathlib import Path
path = Path('main.py')
text = path.read_text(encoding='utf-8')
pattern = re.compile(r'def clean_text\(value: object\) -> Optional\[str\]:[\s\S]+?return text\n\n', re.MULTILINE)
replacement = '''def clean_text(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).replace("\\n", " ").replace("\\r", " ").replace("\\xa0", " ").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("dipnot") or lowered.startswith("not"):
        return None
    if lowered in {"toplam-total", "toplam", "total"}:
        return "Turkiye"
    replacement_map = str.maketrans({
        "\u0130": "I",
        "\u0131": "i",
        "\u015E": "S",
        "\u015F": "s",
        "\u00DC": "U",
        "\u00FC": "u",
        "\u00D6": "O",
        "\u00F6": "o",
        "\u00C7": "C",
        "\u00E7": "c",
        "\uFFFD": "",
        "\u01EC": "u",
    })
    text = text.translate(replacement_map)
    return text

'''
if not pattern.search(text):
    raise SystemExit('clean_text pattern not found')
text = pattern.sub(replacement, text, count=1)
path.write_text(text, encoding='utf-8')
