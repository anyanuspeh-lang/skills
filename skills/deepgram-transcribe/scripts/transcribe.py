#!/usr/bin/env python3
"""
Deepgram Transcription CLI
Transcribes local audio/video files or remote URLs (YouTube, Podcasts, etc.)
using Deepgram's Nova-3 model with speaker diarization and timestamps.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
import urllib.request
import urllib.parse
from pathlib import Path

DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


def get_api_key():
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        # Fallback to configured key
        key = "3ebf872ddf01493f19908ef56ec30a182726ad9f"
    return key


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    m = int(m % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def download_audio_from_url(url: str, output_path: str) -> str:
    print(f"📥 Загрузка аудиопотока по ссылке через yt-dlp: {url}...")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "4",
        "-o", output_path,
        "--no-playlist",
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Ошибка при скачивании аудио: {res.stderr}")
    return output_path


def extract_audio_with_ffmpeg(input_path: str, output_path: str) -> str:
    print(f"🎵 Извлечение аудиодорожки через ffmpeg: {input_path}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "4",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Ошибка ffmpeg: {res.stderr}")
    return output_path


def transcribe_audio(file_path: str, api_key: str, model="nova-3", language="ru", detect_language=False) -> dict:
    params = {
        "model": model,
        "smart_format": "true",
        "diarize": "true",
        "punctuate": "true",
        "paragraphs": "true"
    }
    if detect_language:
        params["detect_language"] = "true"
    elif language:
        params["language"] = language

    query_str = urllib.parse.urlencode(params)
    endpoint = f"{DEEPGRAM_API_URL}?{query_str}"

    print(f"⚡ Отправка в Deepgram (модель: {model})...")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"📦 Размер файла: {file_size_mb:.2f} МБ")

    with open(file_path, "rb") as f:
        audio_data = f.read()

    req = urllib.request.Request(
        endpoint,
        data=audio_data,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/mp3"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        raise RuntimeError(f"Ошибка Deepgram API ({e.code}): {err_body}")


def format_transcript_markdown(result: dict) -> str:
    lines = []
    metadata = result.get("metadata", {})
    duration = metadata.get("duration", 0)
    lines.append(f"# Транскрипция аудио\n")
    lines.append(f"- **Длительность:** {format_timestamp(duration)}")
    lines.append(f"- **Модель:** Deepgram Nova")
    lines.append("")
    lines.append("## Текст с таймкодами и спикерами\n")

    channels = result.get("results", {}).get("channels", [])
    if not channels:
        return "Текст не найден."

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return "Текст не найден."

    alt = alternatives[0]
    paragraphs_data = alt.get("paragraphs", {}).get("paragraphs", [])

    if paragraphs_data:
        for p in paragraphs_data:
            speaker = p.get("speaker", 0)
            start_time = p.get("start", 0)
            time_str = format_timestamp(start_time)
            sentences = [s.get("text", "") for s in p.get("sentences", [])]
            text = " ".join(sentences).strip()
            lines.append(f"**[{time_str}] Спикер {speaker}:**")
            lines.append(f"{text}\n")
    else:
        # Fallback to full transcript
        full_transcript = alt.get("transcript", "")
        lines.append(full_transcript)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Deepgram Audio/Video Transcription Tool")
    parser.add_argument("source", help="Путь к аудио/видео файлу или URL (YouTube, Rutube, VK и др.)")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата (.md или .txt)")
    parser.add_argument("-m", "--model", default="nova-3", help="Модель Deepgram (nova-3, nova-2)")
    parser.add_argument("-l", "--language", default="ru", help="Язык аудио (ru, en или 'auto' для автоопределения)")

    args = parser.parse_args()
    api_key = get_api_key()

    source = args.source
    is_url = source.startswith("http://") or source.startswith("https://")
    detect_lang = (args.language.lower() == "auto")
    lang = None if detect_lang else args.language

    with tempfile.TemporaryDirectory() as tmpdir:
        target_audio = os.path.join(tmpdir, "audio.mp3")

        if is_url:
            download_audio_from_url(source, target_audio)
        else:
            ext = Path(source).suffix.lower()
            if ext in [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]:
                extract_audio_with_ffmpeg(source, target_audio)
            else:
                target_audio = source

        result_json = transcribe_audio(
            target_audio,
            api_key=api_key,
            model=args.model,
            language=lang,
            detect_language=detect_lang
        )

        md_output = format_transcript_markdown(result_json)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md_output)
            print(f"✅ Транскрипция успешно сохранена в: {args.output}")
        else:
            print("\n" + "=" * 50)
            print(md_output)
            print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
