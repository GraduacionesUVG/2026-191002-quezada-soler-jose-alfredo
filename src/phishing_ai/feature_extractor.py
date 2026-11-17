import re
import urllib.parse
from typing import List

SUSPICIOUS_WORDS = [
    'login', 'verify', 'secure', 'update', 'confirm', 'account',
    'banking', 'signin', 'password', 'credential', 'wallet', 'paypal',
    'ebay', 'amazon', 'microsoft', 'apple', 'bank', 'free', 'prize',
    'winner', 'click', 'urgent', 'suspended', 'limited', 'validate'
]

FEATURE_NAMES = [
    'length_url', 'count_dots', 'has_https', 'has_at_symbol',
    'has_ip_address', 'count_hyphens', 'suspicious_words_count',
    'count_subdomains', 'count_slashes', 'count_question_marks',
    'count_equal_signs', 'count_underscores', 'count_percent',
    'length_domain', 'count_digits_url'
]


def extract_features(url: str) -> List[float]:
    url_lower = url.lower()

    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ''
    except Exception:
        hostname = ''

    features = [
        len(url),
        url.count('.'),
        1.0 if url_lower.startswith('https://') else 0.0,
        1.0 if '@' in url else 0.0,
        1.0 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', hostname) else 0.0,
        float(url.count('-')),
        float(sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)),
        float(max(0, len(hostname.split('.')) - 2)),
        float(url.count('/')),
        float(url.count('?')),
        float(url.count('=')),
        float(url.count('_')),
        float(url.count('%')),
        float(len(hostname)),
        float(sum(c.isdigit() for c in url)),
    ]

    return features
