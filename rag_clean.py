"""
RAG Product Data Cleaner
========================
Transforms cleaned_products.json → rag_products.json
Optimized for retrieval-augmented generation embedding.

Decisions applied:
- Colors: Keep named colors as-is, keep hex as-is, drop junk #000000 arrays
- Shipping/Return: Dropped (brand-level, not product-level)
- Product Info + Specs: Merged into single flat specifications string
- Dynamic variants: Alongside size/color as separate attribute fields
- Weight: Moved to attributes
- Discounted price: Kept for future use even though currently all equal original
- Output: rag_products.json (new file)
"""

import json
import re
from html import unescape


INPUT_FILE  = '/home/mush/git/github/numobel/cleaned_products.json'
OUTPUT_FILE = '/home/mush/git/github/numobel/rag_products.json'


# ──────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────

def clean_html(text):
    """Strip HTML tags, unescape entities, collapse whitespace."""
    if not isinstance(text, str):
        return None
    text = re.sub(r'<(br|p|/p)[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None


def norm_str(s):
    """Strip a string, return None if empty."""
    if isinstance(s, str):
        s = s.strip()
        return s if s else None
    return None


def dedup_list(lst):
    """Deduplicate a list preserving order."""
    seen = set()
    out = []
    for item in (lst or []):
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ──────────────────────────────────────────────
# Color cleaning
# ──────────────────────────────────────────────

def clean_colors(colors):
    """
    Keep named colors as-is, keep hex as-is.
    Drop junk arrays where ALL values are #000000.
    """
    if not colors or not isinstance(colors, list):
        return []

    # Check if the entire array is junk (all #000000)
    if all(c == '#000000' for c in colors):
        return []

    # Filter out individual #000000 entries if mixed with real values
    cleaned = [c for c in colors if c != '#000000']
    return dedup_list(cleaned)


# ──────────────────────────────────────────────
# SEO extraction
# ──────────────────────────────────────────────

def extract_seo(product):
    """Parse Wix SEO tag objects and keyword objects into flat arrays."""
    raw_tags = product.get('seo_tag') or []
    raw_keywords = product.get('seo_keywords') or []

    tags = []
    keywords = []

    # SEO tags: extract meta description content and other tag values
    for tag in raw_tags:
        if isinstance(tag, dict):
            props = tag.get('props', {})
            content = norm_str(props.get('content', ''))
            name = (props.get('name') or '').lower()
            if name == 'keywords' and content:
                # Split comma-separated keywords
                keywords.extend([k.strip().lower() for k in content.split(',') if k.strip()])
            elif content:
                tags.append(content)
        elif isinstance(tag, str) and tag.strip():
            tags.append(tag.strip())

    # SEO keywords: parse {term, isMain} objects
    for kw in raw_keywords:
        if isinstance(kw, dict):
            term = norm_str(kw.get('term', ''))
            if term:
                keywords.append(term.lower())
        elif isinstance(kw, str) and kw.strip():
            keywords.append(kw.strip().lower())

    # Deduplicate
    tags = dedup_list(tags)
    keywords = dedup_list(keywords)

    return tags, keywords


# ──────────────────────────────────────────────
# Specifications merging
# ──────────────────────────────────────────────

def merge_specifications(product):
    """
    Merge product_info + specifications into a single flat string.
    Format: "Key: Value; Key: Value; ..."
    Deduplicates keys (product_info wins if conflict).
    """
    merged = {}

    # product_info first (general info)
    pi = product.get('product_info')
    if isinstance(pi, dict):
        for k, v in pi.items():
            v_str = str(v).strip() if v else ''
            if v_str:
                merged[k] = v_str

    # specifications second (technical specs)
    sp = product.get('specifications')
    if isinstance(sp, dict):
        for k, v in sp.items():
            v_str = str(v).strip() if v else ''
            if v_str and k not in merged:  # don't overwrite product_info
                merged[k] = v_str

    if not merged:
        return None

    return '; '.join(f'{k}: {v}' for k, v in merged.items())


# ──────────────────────────────────────────────
# Price extraction
# ──────────────────────────────────────────────

def extract_price(product):
    """
    Return (original, discounted).
    Keep both even if identical (for future discount support).
    """
    original = product.get('price')
    discounted = product.get('discounted_price')

    # Normalize types
    if isinstance(original, (int, float)):
        original = float(original)
    else:
        original = None

    if isinstance(discounted, (int, float)):
        discounted = float(discounted)
    else:
        discounted = None

    return original, discounted


# ──────────────────────────────────────────────
# Image cleaning
# ──────────────────────────────────────────────

def clean_images(links):
    """Deduplicate image URLs; ensure they're valid HTTP(S)."""
    if not links or not isinstance(links, list):
        return []
    cleaned = []
    seen = set()
    for url in links:
        if isinstance(url, str) and url.startswith('http') and url not in seen:
            seen.add(url)
            cleaned.append(url)
    return cleaned


# ──────────────────────────────────────────────
# Main transform
# ──────────────────────────────────────────────

# Dynamic variant keys to capture alongside size/color
VARIANT_KEYS = ['partition_thickness', 'pull_type', 'worktop_size']

def infer_brand(product):
    """
    Infer brand from explicit CSV field or from product name prefix.
    5 brands: Rubio Monocoat, Nuacoustics, Nupanel, Nuwork, Nutoy.
    Fallback: Numobel (generic parent brand).
    Handles known typos in source data.
    """
    brand = norm_str(product.get('brand'))
    # If the explicit brand is a generic parent ('Numobel'), still run name-based
    # detection — sub-brands (Nupanel, Nuwork, Nuacoustics) are stored as 'Numobel'
    # in the source CSV and need to be reclassified from the product name.
    if brand and brand != 'Numobel':
        return brand

    name = (product.get('name') or '').strip()
    # Order matters: more specific prefixes first to avoid greedy fallback.
    brand_prefixes = [
        # Rubio Monocoat (including typo 'Rubio Moonocoat')
        ('Rubio Monocoat',    'Rubio Monocoat'),
        ('Rubio Moonocoat',   'Rubio Monocoat'),

        # Nuacoustics — check all casing/typo variants before generic Numobel
        ('Numobel Acoustics', 'Nuacoustics'),
        ('Numobel acoustics', 'Nuacoustics'),   # lowercase typo
        ('Numoble Acoustics', 'Nuacoustics'),   # spelling typo

        # Nupanel — Open Work Panel products are named OWP-XX
        ('OWP',               'Nupanel'),

        # Nuwork — workstation systems and mobile storage
        ('Workstation',       'Nuwork'),
        ('Storage Closed',    'Nuwork'),

        # Nutoy (including 'Numobel-Toys-...' naming variant)
        ('Nutoy',             'Nutoy'),
        ('Numobel-Toys',      'Nutoy'),

        # Generic Numobel fallback
        ('Numobel',           'Numobel'),
    ]
    for prefix, brand_name in brand_prefixes:
        if name.startswith(prefix) or name.lower().startswith(prefix.lower()):
            return brand_name

    return None


# ──────────────────────────────────────────────
# Product line inference
# ──────────────────────────────────────────────

# Nutoy: map segment-at-index-1 to canonical product line
_NUTOY_LINE_MAP = {
    'On Wheels':                  'On Wheels',
    'Stacker':                    'Stacker',
    'Montessori':                 'Montessori',
    'Montessori Ball Tracker':    'Montessori',
    'Montessori Object Permanence Box Mini':     'Montessori',
    'Montessori Object Permanence Box with Drawer': 'Montessori',
    'Waldorf':                    'Building Block',
    'Waldorf Vehicles':           'Building Block',
    'Building Block':             'Building Block',
    'Wooden':                     'Building Block',   # Domino sets
    'Balancing':                  'Balancing',
    'Chinese':                    'Board Games',
    'Puzzle':                     'Learning',
    'Puzzle Geometric':           'Learning',
    'Learning':                   'Learning',
    'Cuboid':                     'Furniture',
    'Kiddo':                      'Furniture',
    'Components':                 'Components',
    'Component':                  'Components',   # singular variant in 'Numobel-Toys-Component-...'
    'Toys':                       None,               # skip; use next segment
}

# Nuacoustics: map segment-at-index-1 to canonical product line
_NUACOUSTICS_LINE_MAP = {
    'PET VG':               'PET VG',
    'PETLight':             'PET Light',
    'PET Acoustic Sheets':  'PET Plain',
    'PET Ceiling Baffle':   'PET Ceiling',
    'PET Ceiling Cloud':    'PET Ceiling',
    'MDF Perforated':       'MDF Perforated',
}


def infer_product_line(brand, name):
    """
    Infer the product line (sub-series) from brand + product name.
    Returns a string or None.
    """
    if not name or not brand:
        return None

    parts = [p.strip() for p in name.split('-')]

    if brand == 'Nutoy':
        # Handle 'Numobel-Toys-Component-...' naming variant:
        # index 0 = 'Numobel', index 1 = 'Toys' (skip), index 2 = 'Component'
        if len(parts) > 1 and parts[1] == 'Toys' and len(parts) > 2:
            seg = parts[2]
        else:
            seg = parts[1] if len(parts) > 1 else None

        if seg is None:
            return None

        # Direct lookup — try progressively shorter matches for multi-word segments
        if seg in _NUTOY_LINE_MAP:
            mapped = _NUTOY_LINE_MAP[seg]
            # None means 'skip this segment' — shouldn't happen outside Toys edge case
            return mapped
        # Partial match (e.g. 'Montessori Ball Tracker' stored as single segment)
        for key, val in _NUTOY_LINE_MAP.items():
            if seg.lower().startswith(key.lower()):
                return val
        return None

    if brand == 'Nuacoustics':
        # Name format: 'Numobel Acoustics-<LINE>-<detail>'
        # After splitting on '-', parts[0] = 'Numobel Acoustics', parts[1] = line
        seg = parts[1] if len(parts) > 1 else None
        if seg is None:
            return None
        # Handle 'MDF Perforated' which may be attached to rest of name
        if seg in _NUACOUSTICS_LINE_MAP:
            return _NUACOUSTICS_LINE_MAP[seg]
        # Partial match for variants
        for key, val in _NUACOUSTICS_LINE_MAP.items():
            if seg.lower().startswith(key.lower()):
                return val
        return None

    if brand == 'Nuwork':
        if name.startswith('Workstation') or name.startswith('Storage Closed'):
            return 'Panelsys' if 'Workstation' in name or 'Panelsys' in name else 'Storage'
        return None

    if brand == 'Nupanel':
        return 'Open Work Panel'

    # Rubio Monocoat and Numobel: no product line structure
    return None


def transform(product):
    """Transform a single product into the RAG-optimized schema."""

    tags, keywords = extract_seo(product)
    original_price, discounted_price = extract_price(product)

    # Build attributes
    attributes = {
        'colors': clean_colors(product.get('colors')),
        'size': dedup_list(product.get('size') or []),
        'weight': norm_str(product.get('weight')),
        'specifications': merge_specifications(product),
    }

    # Add dynamic variant keys
    for vk in VARIANT_KEYS:
        val = product.get(vk)
        if isinstance(val, list):
            attributes[vk] = dedup_list(val)
        else:
            attributes[vk] = []

    brand = infer_brand(product)
    name  = norm_str(product.get('name'))

    return {
        'name': name,
        'description': clean_html(product.get('product_description')),
        'brand': brand,
        'product_line': infer_product_line(brand, name),
        'price': {
            'original': original_price,
            'discounted': discounted_price,
        },
        'seo': {
            'tags': tags,
            'keywords': keywords,
        },
        'attributes': attributes,
        'media': {
            'images': clean_images(product.get('product_image_links')),
        },
        'metadata': {
            'sku': norm_str(product.get('sku')),
            'product_link': norm_str(product.get('product_link')),
        },
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    with open(INPUT_FILE, encoding='utf-8') as f:
        products = json.load(f)

    cleaned = [transform(p) for p in products]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # Verification
    keys_set = set(tuple(sorted(p.keys())) for p in cleaned)
    print(f'Processed {len(cleaned)} products → {OUTPUT_FILE}')
    print(f'Schema variants: {len(keys_set)} (should be 1)')

    # File size comparison
    import os
    in_size  = os.path.getsize(INPUT_FILE)
    out_size = os.path.getsize(OUTPUT_FILE)
    print(f'Size: {in_size:,} bytes → {out_size:,} bytes ({100 - out_size*100//in_size}% reduction)')


if __name__ == '__main__':
    main()
