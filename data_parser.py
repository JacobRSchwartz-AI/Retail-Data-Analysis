import json
import os
import csv
from typing import List, Dict

def parse_json(file_name):
    file_path = os.path.join('data', file_name)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Remove the second and second to last characters (the quotes around the array)
        content = content[0] + content[2:-2] + content[-1]
        
        # Fix all the escaping issues 
        content = content.replace('\\n', '\n')
        content = content.replace('\\"', '"')
        
        # Fix the object separation issue
        content = content.replace('}","{', '},{')
        
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            print(f"Error: {str(e)}")
            print("Content:", content)
            return None

def extract_purchases(data_dir: str = 'data') -> List[Dict]:
    all_purchases = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    for file_name in sorted(json_files):
        data = parse_json(file_name)
        if data:
            for customer in data:
                for purchase in customer['purchases']:
                    purchase_row = {
                        'customerId': purchase['customerId'],
                        'productId': purchase['productId'],
                        'productName': purchase['productName'],
                        'productCategory': purchase['productCategory'],
                        'purchaseAmount': purchase['purchaseAmount'],
                        'purchaseDate': purchase['purchaseDate']
                    }
                    all_purchases.append(purchase_row)
    
    return all_purchases

def save_to_csv(purchases: List[Dict], output_file: str = 'purchases.csv'):
    if not purchases:
        print("No purchases to write to CSV")
        return
    
    fieldnames = ['customerId', 'productId', 'productName', 'productCategory', 'purchaseAmount', 'purchaseDate']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(purchases)

# Execute the process
purchases = extract_purchases()
save_to_csv(purchases)