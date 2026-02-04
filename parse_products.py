import csv
import json
import re
from html import unescape

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return None
    cleanr = re.compile('<.*?>')
    # convert <br> and <p> to spaces to not join words
    raw_html = re.sub(r'<(br|p|/p)[^>]*>', ' ', raw_html, flags=re.IGNORECASE)
    # clean other html tags
    cleantext = re.sub(cleanr, '', raw_html)
    # unescape html entities like &nbsp;
    cleantext = unescape(cleantext)
    # convert multiple spaces to a single space
    return " ".join(cleantext.split())

def to_snake_case(name):
    if not name:
        return "unknown"
    # replace spaces and non-word chars with underscores
    name = re.sub(r'[^\w]+', '_', name)
    # add underscores before capitals if previous char is lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    # collapse multiple underscores
    return re.sub(r'_+', '_', s2).strip('_')

def process_csv(input_file, output_file):
    # Using utf-8-sig to automatically handle any BOMs at the start of the file
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        products = []
        for row in reader:
            product = {}
            
            # Explicit mappings requested
            product['name'] = row.get('Name') or None
            product['product_link'] = row.get('Product Page Url') or None
            try:
                product['price'] = float(row.get('Price')) if row.get('Price') else None
            except ValueError:
                product['price'] = row.get('Price')
            product['weight'] = row.get('Weight') or None
            product['product_description'] = clean_html(row.get('Description'))
            
            # Extract SEO Data
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
                    
            # Extract Product Options
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
                        # Remove None and empty string
                        values = [v for v in values if v]
                        if name == 'color' or name == 'colour':
                            product['colors'] = values
                        elif name == 'size':
                            product['size'] = values
                        else:
                            # dynamically add other variation types
                            product[to_snake_case(name)] = values
                except Exception:
                    pass
                    
            # Extract Product Image Links
            media_str = row.get('Media Items')
            product['product_image_links'] = []
            if media_str:
                try:
                    media_items = json.loads(media_str)
                    for item in media_items:
                        src = item.get('src')
                        if src:
                            product['product_image_links'].append(src)
                except Exception:
                    pass
            if not product['product_image_links']:
                main_media = row.get('Main Media')
                if main_media:
                    product['product_image_links'].append(main_media)

            # Extract Additional Info Sections
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
                        
                        # Helper to parse tables
                        parsed_dict = {}
                        table_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', desc, re.IGNORECASE | re.DOTALL)
                        for tr in table_rows:
                            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, re.IGNORECASE | re.DOTALL)
                            if len(cells) == 2:
                                key = clean_html(cells[0])
                                key = re.sub(r':$', '', key).strip()
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
                            # Catch-all for any other sections
                            snake_title = to_snake_case(title)
                            product[snake_title] = parsed_dict if parsed_dict else (cleaned_desc if cleaned_desc else None)
                except Exception:
                    pass

            # Add other CSV fields explicitly normalizing to snake_case
            mapped_columns = {
                'Name', 'Product Page Url', 'Price', 'Weight', 'Description', 
                'SEO Data', 'Product Options', 'Media Items', 'Main Media', 
                'Additional Info Sections'
            }
            for col in row:
                if col not in mapped_columns:
                    snake_col = to_snake_case(col)
                    val = row[col]
                    if val == '':
                        product[snake_col] = None
                    else:
                        # try to parse as JSON if looks like array or object, useful for embedded json lists
                        if val and ((val.startswith('[') and val.endswith(']')) or (val.startswith('{') and val.endswith('}'))):
                            try:
                                product[snake_col] = json.loads(val)
                            except Exception:
                                product[snake_col] = val
                        else:
                            # Normalize boolean-like strings
                            if val.lower() == 'true':
                                product[snake_col] = True
                            elif val.lower() == 'false':
                                product[snake_col] = False
                            elif val.isdigit():
                                try:
                                    product[snake_col] = int(val)
                                except ValueError:
                                    pass
                            else:
                                product[snake_col] = val
                    
            products.append(product)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=4, ensure_ascii=False)
        print(f"Successfully processed {len(products)} products and saved to {output_file}")

if __name__ == '__main__':
    process_csv('/home/mush/git/github/numobel/Products.csv', '/home/mush/git/github/numobel/cleaned_products.json')
