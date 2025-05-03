import requests
import json
from time import sleep

def test_api():
    """Tests the API with the subtlety of a sledgehammer."""
    API_URL = "http://localhost:8001/recommend"
    
    # Test cases that should work (in theory)
    test_cases = [
        {
            "name": "Technical Test - Java Developer",
            "query": "I need a Java coding test for mid-level developers, should take about 45 minutes",
            "expected_types": ["K", "T"],  # Knowledge or Technical
            "max_duration": 45
        },
        {
            "name": "Behavioral Test - COO Cultural Fit",
            "query": "I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour",
            "expected_types": ["B", "P"],  # Behavioral or Personality
            "max_duration": 60
        },
        {
            "name": "Vague HR Request",
            "query": "We need something for new graduates, maybe with teamwork?",
            "expected_types": ["B"],
            "max_duration": None
        },
        {
            "name": "Broken Query",
            "query": "asdfghjkl",
            "expected_types": [],
            "max_duration": None,
            "should_fail": True
        }
    ]

    for case in test_cases:
        print(f"\n=== Testing: {case['name']} ===")
        print(f"Query: '{case['query']}'")
        
        try:
            response = requests.post(
                API_URL,
                json={"query": case['query']},
                timeout=10
            )
            
            if case.get('should_fail'):
                if response.status_code == 200:
                    print("❌ FAIL: This should have failed but didn't")
                else:
                    print(f"✅ PASS (failed as expected): {response.status_code}")
                continue
                
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                continue
                
            results = response.json().get('recommendations', [])
            
            if not results:
                print("❌ No recommendations returned")
                continue
                
            print(f"✅ Got {len(results)} recommendations")
            
            # Check if any results match expected types
            if case['expected_types']:
                valid_types = any(
                    any(t in rec['test_type'] for t in case['expected_types'])
                    for rec in results
                )
                print(f"   - Contains expected types: {'✅' if valid_types else '❌'}")
            
            # Check duration if specified
            if case['max_duration']:
                within_duration = all(
                    abs(rec['duration'] - case['max_duration']) <= 15
                    for rec in results
                )
                print(f"   - Within duration tolerance: {'✅' if within_duration else '❌'}")
            
            # Print first recommendation details
            print("\nFirst recommendation:")
            first = results[0]
            print(f"   - Name: {first['name']}")
            print(f"   - Duration: {first['duration']} mins")
            print(f"   - Types: {first['test_type']}")
            print(f"   - URL: {first['url']}")
            
        except Exception as e:
            print(f"❌ Test crashed: {str(e)}")
        
        sleep(1)  # Be kind to your API

if __name__ == "__main__":
    print("Starting API tests...")
    test_api()
    print("\nTests complete. Now go fix whatever broke.")