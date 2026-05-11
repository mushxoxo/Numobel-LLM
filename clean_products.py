"""
clean_products.py
===================
Single-step pipeline: Products.csv → rag_products.json

Combines the logic of parse_products.py (CSV → intermediate dict)
and rag_clean.py (intermediate dict → RAG-optimised schema).
"""

import csv
import json
import logging
import re
import os
from html import unescape


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'Products.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'clean_products.json')

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Shared text helpers
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


def to_snake_case(name):
    if not name:
        return 'unknown'
    name = re.sub(r'[^\w]+', '_', name)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return re.sub(r'_+', '_', s2).strip('_')


def extract_wix_image_url(wix_url):
    if not wix_url or not isinstance(wix_url, str):
        return wix_url
    match = re.search(r'wix:image://v1/([^/]+)', wix_url)
    if match:
        return f"https://static.wixstatic.com/media/{match.group(1)}"
    return wix_url


# ──────────────────────────────────────────────
# Stage 1: CSV row → intermediate product dict
# (was parse_products.py)
# ──────────────────────────────────────────────

def _parse_row(row):
    """Parse a single CSV row into an intermediate product dict."""
    product = {}

    product['name'] = row.get('Name') or None

    product_link = row.get('Product Page Url')
    if product_link:
        product_link = product_link.strip('/')
        if product_link.startswith('product-page/'):
            product_link = product_link[len('product-page/'):]
        product['product_link'] = f"https://www.numobel.in/product-page/{product_link}"
    else:
        product['product_link'] = None

    try:
        product['price'] = float(row.get('Price')) if row.get('Price') else None
    except ValueError:
        product['price'] = row.get('Price')

    product['weight'] = row.get('Weight') or None
    product['product_description'] = clean_html(row.get('Description'))

    # SEO data
    seo_data_str = row.get('SEO Data')
    product['seo_tag'] = []
    product['seo_content'] = None
    product['seo_keywords'] = []
    if seo_data_str:
        try:
            seo_data = json.loads(seo_data_str)
            product['seo_tag'] = seo_data.get('tags', [])
            if 'settings' in seo_data:
                product['seo_keywords'] = seo_data['settings'].get('keywords', [])
        except Exception:
            pass

    # Product options (colors, sizes, dynamic variants)
    options_str = row.get('Product Options')
    product['colors'] = []
    product['size'] = []
    if options_str:
        try:
            options = json.loads(options_str)
            for key, val in options.items():
                name = val.get('name', '').lower()
                choices = val.get('choices', [])
                values = [c.get('value') or c.get('description') for c in choices]
                values = [v for v in values if v]
                if name in ('color', 'colour'):
                    product['colors'] = values
                elif name == 'size':
                    product['size'] = values
                else:
                    product[to_snake_case(name)] = values
        except Exception:
            pass

    # Image links
    media_str = row.get('Media Items')
    product['product_image_links'] = []
    if media_str:
        try:
            media_items = json.loads(media_str)
            for item in media_items:
                src = item.get('src')
                if src:
                    product['product_image_links'].append(extract_wix_image_url(src))
        except Exception:
            pass
    if not product['product_image_links']:
        main_media = row.get('Main Media')
        if main_media:
            product['product_image_links'].append(extract_wix_image_url(main_media))

    # Additional info sections
    additional_info_str = row.get('Additional Info Sections')
    product['product_info'] = None
    product['specifications'] = None
    product['shipping_info'] = None
    product['return_and_refund_policy'] = None

    if additional_info_str:
        try:
            add_info = json.loads(additional_info_str)
            for section in add_info:
                title = section.get('title', '').strip().upper()
                desc = section.get('description', '')
                cleaned_desc = clean_html(desc)

                parsed_dict = {}
                table_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', desc, re.IGNORECASE | re.DOTALL)
                for tr in table_rows:
                    cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, re.IGNORECASE | re.DOTALL)
                    if len(cells) == 2:
                        key = clean_html(cells[0])
                        key = re.sub(r':$', '', key).strip() if key else ''
                        value = clean_html(cells[1])
                        if key:
                            parsed_dict[key] = value

                if title == 'PRODUCT INFO':
                    product['product_info'] = parsed_dict if parsed_dict else (cleaned_desc if cleaned_desc else None)
                elif title == 'SPECIFICATIONS':
                    product['specifications'] = parsed_dict if parsed_dict else (cleaned_desc if cleaned_desc else None)
                elif title == 'SHIPPING INFO':
                    product['shipping_info'] = cleaned_desc if cleaned_desc else None
                elif title == 'RETURN & REFUND POLICY':
                    product['return_and_refund_policy'] = cleaned_desc if cleaned_desc else None
                else:
                    snake_title = to_snake_case(title)
                    product[snake_title] = parsed_dict if parsed_dict else (cleaned_desc if cleaned_desc else None)
        except Exception:
            pass

    # Remaining CSV columns (not yet mapped)
    mapped_columns = {
        'Name', 'Product Page Url', 'Price', 'Weight', 'Description',
        'SEO Data', 'Product Options', 'Media Items', 'Main Media',
        'Additional Info Sections'
    }
    for col in row:
        if col not in mapped_columns:
            snake_col = to_snake_case(str(col)) if col else 'extra_fields'
            val = row[col]
            if val == '' or val is None:
                product[snake_col] = None
            elif isinstance(val, str) and (
                (val.startswith('[') and val.endswith(']')) or
                (val.startswith('{') and val.endswith('}'))
            ):
                try:
                    product[snake_col] = json.loads(val)
                except Exception:
                    product[snake_col] = val
            else:
                if isinstance(val, str) and val.lower() == 'true':
                    product[snake_col] = True
                elif isinstance(val, str) and val.lower() == 'false':
                    product[snake_col] = False
                elif isinstance(val, str) and val.isdigit():
                    try:
                        product[snake_col] = int(val)
                    except ValueError:
                        pass
                else:
                    product[snake_col] = val

    return product


def parse_csv(input_file):
    """Read Products.csv and return a list of intermediate product dicts."""
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return [_parse_row(row) for row in reader]


# ──────────────────────────────────────────────
# Stage 2: intermediate dict → RAG schema
# (was rag_clean.py)
# ──────────────────────────────────────────────

def _clean_colors(colors):
    if not colors or not isinstance(colors, list):
        return []
    if all(c == '#000000' for c in colors):
        return []
    cleaned = [c for c in colors if c != '#000000']
    return dedup_list(cleaned)


def _extract_seo(product):
    raw_tags = product.get('seo_tag') or []
    raw_keywords = product.get('seo_keywords') or []

    tags = []
    keywords = []

    for tag in raw_tags:
        if isinstance(tag, dict):
            props = tag.get('props', {})
            content = norm_str(props.get('content', ''))
            name = (props.get('name') or '').lower()
            if name == 'keywords' and content:
                keywords.extend([k.strip().lower() for k in content.split(',') if k.strip()])
            elif content:
                tags.append(content)
        elif isinstance(tag, str) and tag.strip():
            tags.append(tag.strip())

    for kw in raw_keywords:
        if isinstance(kw, dict):
            term = norm_str(kw.get('term', ''))
            if term:
                keywords.append(term.lower())
        elif isinstance(kw, str) and kw.strip():
            keywords.append(kw.strip().lower())

    return dedup_list(tags), dedup_list(keywords)


def _merge_specifications(product):
    merged = {}

    pi = product.get('product_info')
    if isinstance(pi, dict):
        for k, v in pi.items():
            v_str = str(v).strip() if v else ''
            if v_str:
                merged[k] = v_str

    sp = product.get('specifications')
    if isinstance(sp, dict):
        for k, v in sp.items():
            v_str = str(v).strip() if v else ''
            if v_str and k not in merged:
                merged[k] = v_str

    if not merged:
        return None

    return '; '.join(f'{k}: {v}' for k, v in merged.items())


def _extract_price(product):
    original   = product.get('price')
    discounted = product.get('discounted_price')

    original   = float(original)   if isinstance(original,   (int, float)) else None
    discounted = float(discounted)  if isinstance(discounted, (int, float)) else None

    return original, discounted


def _clean_images(links):
    if not links or not isinstance(links, list):
        return []
    cleaned = []
    seen = set()
    for url in links:
        if isinstance(url, str) and url.startswith('http') and url not in seen:
            seen.add(url)
            cleaned.append(url)
    return cleaned


_NUTOY_LINE_MAP = {
    'On Wheels':                                    'On Wheels',
    'Stacker':                                      'Stacker',
    'Montessori':                                   'Montessori',
    'Montessori Ball Tracker':                      'Montessori',
    'Montessori Object Permanence Box Mini':        'Montessori',
    'Montessori Object Permanence Box with Drawer': 'Montessori',
    'Waldorf':                                      'Building Block',
    'Waldorf Vehicles':                             'Building Block',
    'Building Block':                               'Building Block',
    'Wooden':                                       'Building Block',
    'Balancing':                                    'Balancing',
    'Chinese':                                      'Board Games',
    'Puzzle':                                       'Learning',
    'Puzzle Geometric':                             'Learning',
    'Learning':                                     'Learning',
    'Cuboid':                                       'Furniture',
    'Kiddo':                                        'Furniture',
    'Components':                                   'Components',
    'Component':                                    'Components',
    'Toys':                                         None,
}

_NUACOUSTICS_LINE_MAP = {
    'PET VG':               'PET VG',
    'PETLight':             'PET Light',
    'PET Acoustic Sheets':  'PET Plain',
    'PET Ceiling Baffle':   'PET Ceiling',
    'PET Ceiling Cloud':    'PET Ceiling',
    'MDF Perforated':       'MDF Perforated',
}

VARIANT_KEYS = ['partition_thickness', 'pull_type', 'worktop_size']


def _infer_brand(product):
    brand = norm_str(product.get('brand'))
    if brand and brand != 'Numobel':
        return brand

    name = (product.get('name') or '').strip()
    brand_prefixes = [
        ('Rubio Monocoat',    'Rubio Monocoat'),
        ('Rubio Moonocoat',   'Rubio Monocoat'),
        ('Numobel Acoustics', 'Nuacoustics'),
        ('Numobel acoustics', 'Nuacoustics'),
        ('Numoble Acoustics', 'Nuacoustics'),
        ('OWP',               'Nupanel'),
        ('Workstation',       'Nuwork'),
        ('Storage Closed',    'Nuwork'),
        ('Nutoy',             'Nutoy'),
        ('Numobel-Toys',      'Nutoy'),
        ('Numobel',           'Numobel'),
    ]
    for prefix, brand_name in brand_prefixes:
        if name.startswith(prefix) or name.lower().startswith(prefix.lower()):
            return brand_name

    return None


def _infer_product_line(brand, name):
    if not name or not brand:
        return None

    parts = [p.strip() for p in name.split('-')]

    if brand == 'Nutoy':
        if len(parts) > 1 and parts[1] == 'Toys' and len(parts) > 2:
            seg = parts[2]
        else:
            seg = parts[1] if len(parts) > 1 else None

        if seg is None:
            return None

        if seg in _NUTOY_LINE_MAP:
            return _NUTOY_LINE_MAP[seg]
        for key, val in _NUTOY_LINE_MAP.items():
            if seg.lower().startswith(key.lower()):
                return val
        return None

    if brand == 'Nuacoustics':
        seg = parts[1] if len(parts) > 1 else None
        if seg is None:
            return None
        if seg in _NUACOUSTICS_LINE_MAP:
            return _NUACOUSTICS_LINE_MAP[seg]
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

    return None


def _transform(product):
    """Transform a single intermediate product dict into the RAG-optimised schema."""
    tags, keywords = _extract_seo(product)
    original_price, discounted_price = _extract_price(product)

    attributes = {
        'colors':         _clean_colors(product.get('colors')),
        'size':           dedup_list(product.get('size') or []),
        'weight':         norm_str(product.get('weight')),
        'specifications': _merge_specifications(product),
    }
    for vk in VARIANT_KEYS:
        val = product.get(vk)
        attributes[vk] = dedup_list(val) if isinstance(val, list) else []

    brand = _infer_brand(product)
    name  = norm_str(product.get('name'))

    return {
        'name':         name,
        'description':  clean_html(product.get('product_description')),
        'brand':        brand,
        'product_line': _infer_product_line(brand, name),
        'price': {
            'original':   original_price,
            'discounted': discounted_price,
        },
        'seo': {
            'tags':     tags,
            'keywords': keywords,
        },
        'attributes': attributes,
        'media': {
            'images': _clean_images(product.get('product_image_links')),
        },
        'metadata': {
            'sku':          norm_str(product.get('sku')),
            'product_link': norm_str(product.get('product_link')),
        },
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("Reading %s", INPUT_FILE)
    intermediates = parse_csv(INPUT_FILE)
    log.info("Parsed %d products from CSV", len(intermediates))

    cleaned = [_transform(p) for p in intermediates]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    keys_set = set(tuple(sorted(p.keys())) for p in cleaned)
    in_size  = os.path.getsize(INPUT_FILE)
    out_size = os.path.getsize(OUTPUT_FILE)
    log.info("Processed %d products → %s", len(cleaned), OUTPUT_FILE)
    log.info("Schema variants: %d (should be 1)", len(keys_set))
    log.info("Size: %d bytes (CSV) → %d bytes (JSON)", in_size, out_size)


if __name__ == '__main__':
    main()
