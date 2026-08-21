import httpx
import asyncio
import json

API_URL = "http://127.0.0.1:8000"

async def test_text_query(query: str, expect_refusal: bool = False):
    print(f"\n--- Testing Text API: '{query}' ---")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/v1/query",
                json={"query": query, "top_k": 5}
            )
            response.raise_for_status()
            data = response.json()
            
            if data["refusal"]:
                print(f"Refusal received (Expected: {expect_refusal}): {data['refusal_reason']}")
            else:
                print(f"Success (Expected refusal: {expect_refusal}): {data['answer']}")
                print(f"Latency: {data['latency']['total_ms']} ms")
        except Exception as e:
            print(f"Request failed: {e}")

async def test_voice_query():
    print(f"\n--- Testing Voice API ---")
    
    # Generate dummy valid wav data
    # (Since we are using DemoSTT in demo_mode without keys, any wav file is fine)
    # 44 bytes is minimum for valid WAV header
    dummy_wav = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    async with httpx.AsyncClient() as client:
        try:
            files = {'audio': ('test.wav', dummy_wav, 'audio/wav')}
            data = {'language': 'hi-IN', 'top_k': '3'}
            response = await client.post(
                f"{API_URL}/api/v1/query/voice",
                data=data,
                files=files
            )
            response.raise_for_status()
            result = response.json()
            print(f"Transcript: {result['transcript']}")
            print(f"Answer: {result['answer']}")
            print(f"Total Latency: {result['latency']['total_ms']} ms")
            print(f"STT Latency: {result['latency']['stt_ms']} ms")
        except Exception as e:
            print(f"Voice Request failed: {e}")

async def main():
    print("Checking health...")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{API_URL}/api/v1/health")
            print("Health:", r.json())
        except Exception as e:
            print(f"Server unreachable. Is uvicorn running? {e}")
            return
            
    # Valid Hindi MSMARCO Query
    await test_text_query("मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?", expect_refusal=False)
    
    # Off-topic guardrail query
    await test_text_query("What is the recipe for chocolate chip cookies?", expect_refusal=True)
    
    # Voice testing
    await test_voice_query()

if __name__ == "__main__":
    asyncio.run(main())
