import argparse
import requests
import base64
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description="Test vLLM endpoints with CLI inputs")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True, help="Model ID (including vision model for image support)")
    parser.add_argument("--text", help="text prompt")
    parser.add_argument("--audio_url", help="URL for audio file")
    parser.add_argument("--image_urls", nargs="+", help="One or more image URLs")
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    print(f"▶️ Testing model: {args.model}\n")

    if args.text:
        prompt = args.text
        comp = client.completions.create(model=args.model, prompt=prompt, max_tokens=50, temperature=0)
        print("✅ Completion:", comp.choices[0].text.strip())
        chat = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        print("💬 Chat:", chat.choices[0].message.content.strip())

    if args.audio_url:
        local_path = "2086-149220-0033.wav"
        resp = requests.get(args.audio_url)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        with open(local_path, "rb") as audio_file:
            resp = client.audio.transcriptions.create(
                    model=args.model,
                    file=audio_file,
                    language="en",
                    timeout=60.0,
                )
            print("🎧 Transcription:", resp.text)

    if args.image_urls:
        content_list = [{"type": "text", "text": "Analyze these images:"}]
        for url in args.image_urls:
            content_list.append({"type": "image_url", "image_url": {"url": url}})
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content_list}],
            max_completion_tokens=200
        )
        print("🖼️ Vision Chat:", resp.choices[0].message.content.strip())

if __name__ == "__main__":
    main()
