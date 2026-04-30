#!/usr/bin/env python3
import sys
import io
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv(override=True)

from transcriber import transcribe_call
from analyzer import analyze_call
from report import print_report, console


def main():
    parser = argparse.ArgumentParser(description="AtzaAI - ניתוח שיחות מוקד")
    parser.add_argument("audio", help="נתיב לקובץ השמע (MP3/WAV/M4A)")
    parser.add_argument("--save", help="שמור את הפלט ל-JSON", metavar="output.json")
    parser.add_argument("--transcript-only", action="store_true", help="הצג רק תמלול ללא ניתוח")
    args = parser.parse_args()

    audio_path = args.audio
    if not Path(audio_path).exists():
        console.print(f"[red]שגיאה: הקובץ '{audio_path}' לא נמצא[/red]")
        sys.exit(1)

    console.print(f"\n[cyan]🎙️  מתמלל את השיחה...[/cyan]")
    transcript_data = transcribe_call(audio_path)
    console.print(f"[green]✅ תמלול הושלם ({int(transcript_data['duration_seconds'])} שניות)[/green]")

    if args.transcript_only:
        console.print("\n[bold]תמלול השיחה:[/bold]")
        for u in transcript_data["utterances"]:
            color = "blue" if u["speaker"] == "נציג" else "white"
            console.print(f"[{color}][{u['speaker']}]:[/{color}] {u['text']}")
        return

    console.print(f"[cyan]🤖 מנתח את השיחה עם Claude...[/cyan]")
    analysis = analyze_call(transcript_data)
    console.print(f"[green]✅ ניתוח הושלם[/green]")

    print_report(analysis, audio_path)

    if args.save:
        output = {"transcript": transcript_data, "analysis": analysis}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        console.print(f"[dim]💾 נשמר ל: {args.save}[/dim]")


if __name__ == "__main__":
    main()
