import os
from backend.config.settings import get_settings

def main():
    print("--- Effective Runtime Configuration ---")
    try:
        settings = get_settings()
        
        print(f"DEMO_MODE: {settings.demo_mode}")
        print(f"STT_PROVIDER: {settings.stt_provider}")
        print(f"LLM_PROVIDER: {settings.llm_provider}")
        
        sarvam_key = bool(settings.sarvam_api_key)
        groq_key = bool(settings.groq_api_key)
        print(f"SARVAM_API_KEY configured: {str(sarvam_key).lower()}")
        print(f"GROQ_API_KEY configured: {str(groq_key).lower()}")
        
    except Exception as e:
        print(f"Configuration Error: {e}")

if __name__ == "__main__":
    main()
