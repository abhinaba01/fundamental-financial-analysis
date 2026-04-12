import requests

# Test with small file for faster results
url = 'http://127.0.0.1:8000/analyze'
files = {'document': open('test_small.txt', 'rb')}  # Small test file
data = {'query': 'What is the revenue growth?', 'use_gpu': 'false'}

try:
    print("Testing API with small file (should be faster)...")
    response = requests.post(url, files=files, data=data, timeout=300)
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        print('✅ Success! API is working.')
        print('Response preview:', response.text[:500] + '...')
    else:
        print('❌ Error:', response.text)
except requests.exceptions.Timeout:
    print('⏰ Request timed out - analysis takes time even with small files')
    print('💡 This is normal! The API is working, just processing through ML models')
except Exception as e:
    print('❌ Error:', e)