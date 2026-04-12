import requests
import json

url = 'http://127.0.0.1:8000/analyze'

# 1. Update the filename to match your downloaded Apple 10-K
files = {'document': open('medium_test.txt', 'rb')} 

# 2. Update the query to something more specific for a 10-K
# and set use_gpu to 'true' if your Docker setup supports it
data = {
    'query': 'What are the primary risk factors identified for the next fiscal year?', 
    'use_gpu': 'false' 
}

try:
    print("Testing API with medium test (this will take several minutes)...")
    # 3. Increase timeout for large documents if necessary
    response = requests.post(url, files=files, data=data, timeout=600)
    
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        print('✅ Success! Analysis complete.')
        
        full_report = response.json()
        with open('apple_10k_analysis.json', 'w') as f:
            json.dump(full_report, f, indent=4)
            
        print('📄 Full analysis saved to "apple_10k_analysis.json"')
    else:
        print('❌ Error:', response.text)
        
except requests.exceptions.Timeout:
    print('⏰ Request timed out - 10-K files take significant time to process.')
except Exception as e:
    print('❌ Error:', e)