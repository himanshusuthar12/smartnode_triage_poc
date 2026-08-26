"""
SmartNode Support Call Triage Pipeline

Flow:
    Audio/Text file
          |
          +--> Audio -> Faster-Whisper transcription
          |
          +--> Optional Hugging Face speaker diarization
          |
          +--> OpenAI classification (when OPENAI_API_KEY + HF_TOKEN exist)
          |
          +--> Otherwise manual/local rule classification
          |
          +--> Append structured results to call_analysis.csv

Run:
    python main.py

Then enter an audio/text file path when prompted.
Type "exit" or press Ctrl+C to stop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None  # type: ignore


# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
OUTPUT_CSV = BASE_DIR / "call_analysis.csv"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Change this in .env if you use another OpenAI model.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

HF_DIARIZATION_MODEL = os.getenv(
    "HF_DIARIZATION_MODEL",
    "pyannote/speaker-diarization-3.1",
)

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".ogg", ".wma", ".webm"}

TEXT_EXTENSIONS = {".txt", ".md", ".csv"}

CSV_COLUMNS = ["processed_at", "source_file", "source_type", "communication_id", "start_time", "end_time", "duration_seconds", "speaker", "category", "confidence", "action_required", "next_action", "reason", "text", "processing_mode"]

# LOGGING / ENVIRONMENT
def configure_logging() -> None:
    """Configure readable console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def load_configuration() -> Tuple[Optional[str], Optional[str]]:
    """
    Load credentials from .env.

    HF_TOKEN is also accepted as HUGGINGFACEHUB_API_TOKEN.
    """
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
    else:
        load_dotenv(override=False)
        logging.warning(".env was not found. Manual/local mode can still run.")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    hf_token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or None
    )

    return openai_key, hf_token


def api_mode_available( openai_key: Optional[str], hf_token: Optional[str]) -> bool:
    """Return True only when both API credentials are available."""
    return bool(openai_key and hf_token)


# INPUT FILE HANDLING

def validate_input_file(file_path: str) -> Path:
    """Validate and normalize the user-provided file path."""
    cleaned = file_path.strip().strip('"').strip("'")

    if not cleaned:
        raise ValueError("File path cannot be empty.")

    path = Path(cleaned).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"The supplied path is not a file: {path}")

    supported = AUDIO_EXTENSIONS | TEXT_EXTENSIONS

    if path.suffix.lower() not in supported:
        supported_text = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported extensions: {supported_text}")

    return path


def detect_source_type(path: Path) -> str:
    """Detect whether the input is audio or text."""
    if path.suffix.lower() in AUDIO_EXTENSIONS:
        return "audio"
    return "text"


# LOCAL SPEECH-TO-TEXT

class TranscriptionEngine:
    """Lazy-loaded Faster-Whisper transcription engine."""

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            logging.info("Loading Faster-Whisper '%s' (%s/%s)...", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
            self._model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            logging.info("Whisper model loaded successfully.")

        return self._model

    def transcribe(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Transcribe audio into timestamped segments."""
        model = self._load_model()

        logging.info("Transcribing: %s", audio_path)
        segments, _ = model.transcribe(str(audio_path), beam_size=5, vad_filter=True, word_timestamps=True)

        transcript: List[Dict[str, Any]] = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            transcript.append({"start": round(float(segment.start), 2), "end": round(float(segment.end), 2), "text": text})

        if not transcript:
            raise RuntimeError("Audio was processed, but no speech/text was extracted.")

        logging.info("Transcription completed: %d segment(s).", len(transcript))
        return transcript


# HUGGING FACE SPEAKER DIARIZATION

class SpeakerDiarizationEngine:
    """Lazy-loaded pyannote speaker diarization engine."""

    def __init__(self, hf_token: str) -> None:
        self.hf_token = hf_token
        self._pipeline: Any = None

    def _load_pipeline(self) -> Any:
        if PyannotePipeline is None:
            raise ImportError("pyannote.audio is not installed. Run: pip install pyannote.audio")

        if self._pipeline is None:
            logging.info("Loading Hugging Face diarization model...")
            self._pipeline = PyannotePipeline.from_pretrained(HF_DIARIZATION_MODEL, use_auth_token=self.hf_token)
            logging.info("Speaker diarization model loaded successfully.")

        return self._pipeline

    def diarize(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Return speaker/time ranges."""
        pipeline = self._load_pipeline()

        logging.info("Running speaker diarization: %s", audio_path)

        diarization = pipeline(str(audio_path))
        speaker_segments: List[Dict[str, Any]] = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(
                {"start": round(float(turn.start), 2), "end": round(float(turn.end), 2), "speaker": str(speaker)}
            )

        logging.info("Diarization completed: %d speaker segment(s).", len(speaker_segments))
        return speaker_segments


# TRANSCRIPT + SPEAKER MERGING

def calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap duration between two time ranges."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def assign_speaker(transcript_segment: Dict[str, Any], speaker_segments: List[Dict[str, Any]]) -> str:
    """Assign the speaker with the greatest timestamp overlap."""
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for speaker_segment in speaker_segments:
        overlap = calculate_overlap(
            transcript_segment["start"],
            transcript_segment["end"],
            speaker_segment["start"],
            speaker_segment["end"],
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker_segment["speaker"]

    return best_speaker


def merge_transcript_with_speakers(
    transcript: List[Dict[str, Any]],
    speaker_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach speaker labels to Whisper transcript segments."""
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "speaker": assign_speaker(segment, speaker_segments),
            "text": segment["text"],
        }
        for segment in transcript
    ]


# COMMUNICATION SEGMENTATION

def build_communications(
    segments: List[Dict[str, Any]],
    max_gap_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """Merge adjacent segments from the same speaker."""
    if not segments:
        return []

    communications: List[Dict[str, Any]] = []

    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "speaker": segments[0]["speaker"],
        "text": segments[0]["text"],
    }

    for segment in segments[1:]:
        gap = segment["start"] - current["end"]
        same_speaker = segment["speaker"] == current["speaker"]

        if same_speaker and gap <= max_gap_seconds:
            current["end"] = segment["end"]
            current["text"] += " " + segment["text"]
        else:
            communications.append(current)
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"],
                "text": segment["text"],
            }

    communications.append(current)

    for index, communication in enumerate(communications, start=1):
        communication["communication_id"] = index
        communication["duration_seconds"] = round(
            communication["end"] - communication["start"],
            2,
        )

    return communications


# TEXT INPUT

def read_text_file(path: Path) -> List[Dict[str, Any]]:
    """Read a text/markdown/CSV file as one communication."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not read '{path}' as UTF-8 text."
        ) from exc

    if not text:
        raise ValueError(f"Text file is empty: {path}")

    return [{"communication_id": 1, "start": 0.0, "end": 0.0, "duration_seconds": 0.0, "speaker": "USER", "text": text}]


# MANUAL / LOCAL CLASSIFICATION

URGENT_PATTERNS = [
    r"\burgent\b",
    r"\bimmediately\b",
    r"\bemergency\b",
    r"\bcritical\b",
    r"\bsparks?\b",
    r"\bfire\b",
    r"\bsmoke\b",
    r"\bshort[\s-]?circuit\b",
    r"\belectric(?:al)?\s+shock\b",
    r"\boverheating\b",
    r"\bhot\s+switch\b",
    r"\bexplosion\b",
    r"\bblocked\s+(?:service|system|production)\b",
]

CLOSED_PATTERNS = [
    r"\bresolved\b",
    r"\bissue\s+is\s+fixed\b",
    r"\bproblem\s+is\s+fixed\b",
    r"\bworking\s+now\b",
    r"\bcompleted\b",
    r"\bno\s+further\s+action\b",
    r"\bconfirmed\b",
    r"\bcancelled\b",
]

OPEN_PATTERNS = [
    r"\bneed\s+(?:help|support|assistance)\b",
    r"\bplease\s+(?:help|check|investigate|call)\b",
    r"\bwaiting\b",
    r"\bpending\b",
    r"\bnot\s+working\b",
    r"\bdoesn't\s+work\b",
    r"\bdoes\s+not\s+work\b",
    r"\bcan\s+someone\b",
    r"\bengineer\b",
    r"\bfollow[\s-]?up\b",
    r"\binvestigate\b",
    r"\bissue\b",
    r"\bproblem\b",
    r"\bunable\b",
]

NEXT_ACTION_BY_CATEGORY = {
    "URGENT": "Immediate agent escalation and priority action.",
    "OPEN": "Agent follow-up/investigation is required.",
    "CLOSED": "No further action required.",
}


def count_matches(text: str, patterns: List[str]) -> int:
    """Count regex matches in normalized text."""
    normalized = text.lower()
    return sum(bool(re.search(pattern, normalized)) for pattern in patterns)


def manual_classify(text: str) -> Dict[str, Any]:
    """
    Rule-based fallback classifier.

    Priority:
        URGENT > OPEN > CLOSED
    """
    urgent_matches = count_matches(text, URGENT_PATTERNS)
    closed_matches = count_matches(text, CLOSED_PATTERNS)
    open_matches = count_matches(text, OPEN_PATTERNS)

    if urgent_matches:
        category = "URGENT"
        confidence = min(0.99, 0.80 + urgent_matches * 0.04)
        reason = "Manual rule matched an urgent/safety/critical condition."

    elif closed_matches and not open_matches:
        category = "CLOSED"
        confidence = min(0.95, 0.78 + closed_matches * 0.05)
        reason = "Manual rule detected a clear resolution/completion condition."

    elif open_matches or not closed_matches:
        category = "OPEN"
        confidence = min(0.90, 0.60 + open_matches * 0.05)
        reason = "Manual rule detected an active, pending, or unresolved request."

    else:
        category = "OPEN"
        confidence = 0.55
        reason = "No strong rule matched; manual review is recommended."

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "reason": reason,
        "action_required": category != "CLOSED",
        "next_action": NEXT_ACTION_BY_CATEGORY[category],
    }


# OPENAI CLASSIFICATION

SYSTEM_PROMPT = """
You are an expert call-center communication classifier.

Classify each communication into exactly one category:

CLOSED:
The issue/request is resolved, completed, confirmed, cancelled, or no further
action is required.

OPEN:
The issue/request is active, pending, waiting for action, requires investigation
or follow-up, or is not yet resolved.

URGENT:
The issue requires immediate/high-priority attention because of safety,
critical failure, business impact, deadline, blocked service, financial impact,
escalation, or explicit urgent/immediate language.

Rules:
- Classify by meaning and context, not only keywords.
- URGENT takes priority when immediate action is required.
- Do not classify something as CLOSED only because the customer says "thank you".
- A promise to investigate/call back/review is OPEN unless it is clearly critical.
- Return valid JSON only.
"""


def create_openai_client(openai_key: str) -> Any:
    """Create an OpenAI client with a useful dependency error."""
    if OpenAI is None:
        raise ImportError(
            "The 'openai' package is not installed. "
            "Run: pip install openai"
        )

    return OpenAI(api_key=openai_key)


def classify_with_openai(
    client: Any,
    communication: Dict[str, Any],
    previous_context: str = "",
) -> Dict[str, Any]:
    """Classify one communication with OpenAI."""
    prompt = f"""
Analyze this incoming communication.

Previous context:
{previous_context[-8000:]}

Speaker:
{communication["speaker"]}

Text:
{communication["text"]}

Return exactly this JSON structure:
{{
  "category": "CLOSED | OPEN | URGENT",
  "confidence": 0.0,
  "reason": "short explanation",
  "action_required": true,
  "next_action": "required action or null"
}}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("OpenAI returned an empty response.")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI returned invalid JSON.") from exc

    return validate_classification(result)


# RESULT VALIDATION

VALID_CATEGORIES = {"CLOSED", "OPEN", "URGENT"}


def validate_classification(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate classifier output."""
    category = str(result.get("category", "OPEN")).upper().strip()

    if category not in VALID_CATEGORIES:
        category = "OPEN"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    action_required = result.get("action_required")

    if not isinstance(action_required, bool):
        action_required = category != "CLOSED"

    next_action = result.get("next_action")

    if next_action in (None, "", "null"):
        next_action = NEXT_ACTION_BY_CATEGORY[category]

    reason = str(result.get("reason", "")).strip()

    if not reason:
        reason = "No reason returned by classifier."

    return {"category": category, "confidence": round(confidence, 4), "reason": reason, "action_required": action_required,
        "next_action": str(next_action)}


# CSV OUTPUT

def build_csv_rows(
    results: List[Dict[str, Any]],
    source_file: Path,
    source_type: str,
    processing_mode: str,
) -> List[Dict[str, Any]]:
    """Convert pipeline results into flat CSV rows."""
    timestamp = datetime.now().isoformat(timespec="seconds")

    return [
        {
            "processed_at": timestamp,
            "source_file": str(source_file),
            "source_type": source_type,
            "communication_id": result["communication_id"],
            "start_time": result["start"],
            "end_time": result["end"],
            "duration_seconds": result["duration_seconds"],
            "speaker": result["speaker"],
            "category": result["category"],
            "confidence": result["confidence"],
            "action_required": result["action_required"],
            "next_action": result["next_action"],
            "reason": result["reason"],
            "text": result["text"],
            "processing_mode": processing_mode,
        }
        for result in results
    ]


def save_to_csv(
    rows: List[Dict[str, Any]],
    output_file: Path = OUTPUT_CSV,
) -> None:
    """Append results to CSV and create headers when needed."""
    if not rows:
        logging.warning("No rows to save.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows, columns=CSV_COLUMNS)
    file_exists = output_file.exists() and output_file.stat().st_size > 0

    dataframe.to_csv(output_file, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")
    logging.info("Saved %d row(s) to: %s", len(dataframe), output_file)


def print_results(results: List[Dict[str, Any]], processing_mode: str) -> None:
    """Print a compact result table."""
    if not results:
        return

    dataframe = pd.DataFrame(
        [
            {
                "ID": result["communication_id"],
                "Speaker": result["speaker"],
                "Category": result["category"],
                "Confidence": result["confidence"],
                "Action": result["action_required"],
                "Text": result["text"],
            }
            for result in results
        ]
    )

    print("\n" + "=" * 100)
    print(f"PROCESSING MODE: {processing_mode}")
    print("=" * 100)
    print(dataframe.to_string(index=False))
    print("=" * 100)


# MAIN APPLICATION

class CallTriageApplication:
    """Manage extraction, OpenAI processing, and manual fallback."""

    def __init__(
        self,
        openai_key: Optional[str],
        hf_token: Optional[str],
    ) -> None:
        self.openai_key = openai_key
        self.hf_token = hf_token

        self.transcriber = TranscriptionEngine()
        self.openai_client: Any = None
        self.diarizer: Optional[SpeakerDiarizationEngine] = None

        if api_mode_available(openai_key, hf_token):
            logging.info("OPENAI_API_KEY + HF_TOKEN detected. OpenAI/Hugging Face mode is available.")
        else:
            logging.warning( "OPENAI_API_KEY and/or HF_TOKEN is missing. Manual/local mode will be used.")

    def extract_communications(self, source_path: Path, source_type: str) -> List[Dict[str, Any]]:
        """
        Extract communications.

        Audio:
            Faster-Whisper -> optional speaker diarization.

        Text:
            Read the text directly.
        """
        if source_type == "text":
            return read_text_file(source_path)

        transcript = self.transcriber.transcribe(source_path)

        # Speaker diarization is helpful but not allowed to stop the application.
        if self.hf_token:
            try:
                if self.diarizer is None:
                    self.diarizer = SpeakerDiarizationEngine(self.hf_token)

                speaker_segments = self.diarizer.diarize(source_path)

                merged = merge_transcript_with_speakers(transcript, speaker_segments)

                communications = build_communications(merged)

                if communications:
                    return communications

            except Exception as exc:
                logging.warning("Speaker diarization failed: %s", exc)
                logging.warning("Continuing with UNKNOWN speaker labels.")

        fallback_segments = [
            {
                **segment,
                "speaker": "UNKNOWN",
            }
            for segment in transcript
        ]

        return build_communications(fallback_segments)

    def classify(
        self,
        communications: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        OpenAI path is attempted only when both keys are configured.

        If OpenAI is unavailable, misconfigured, returns an error, or the
        response cannot be parsed, the application automatically switches to
        manual/local rules for the complete file.
        """
        if not api_mode_available(self.openai_key, self.hf_token):
            return self._manual_classification(communications)

        try:
            if self.openai_client is None:
                self.openai_client = create_openai_client(
                    self.openai_key  # type: ignore[arg-type]
                )

            results: List[Dict[str, Any]] = []
            context = ""

            for communication in communications:
                logging.info(
                    "OpenAI classifying communication %s...",
                    communication["communication_id"],
                )

                classification = classify_with_openai(
                    self.openai_client,
                    communication,
                    context,
                )

                result = {
                    **communication,
                    **classification,
                }

                results.append(result)

                context += (
                    f"\n{communication['speaker']}: "
                    f"{communication['text']}\n"
                    f"Category: {classification['category']}\n"
                )

            return results, "OPENAI + HUGGING FACE"

        except Exception as exc:
            logging.error("OpenAI path failed: %s", exc)
            logging.warning(
                "Switching to manual/local classification for this file."
            )

            return self._manual_classification(communications)

    @staticmethod
    def _manual_classification(
        communications: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Classify all communications with local rules."""
        logging.info("Running manual/local classification...")

        results: List[Dict[str, Any]] = []

        for communication in communications:
            classification = manual_classify(communication["text"])

            results.append(
                {
                    **communication,
                    **classification,
                }
            )

        return results, "MANUAL / LOCAL RULES"

    def process_file(self, source_path: Path) -> None:
        """Process one file and append its results to CSV."""
        source_type = detect_source_type(source_path)

        logging.info("Processing file: %s", source_path)
        logging.info("Input type: %s", source_type)

        try:
            communications = self.extract_communications(
                source_path,
                source_type,
            )

            if not communications:
                raise RuntimeError(
                    "No communication data was extracted from the input."
                )

            results, processing_mode = self.classify(communications)

            rows = build_csv_rows(
                results,
                source_path,
                source_type,
                processing_mode,
            )

            save_to_csv(rows)
            print_results(results, processing_mode)

        except Exception as exc:
            logging.exception(
                "Failed to process '%s': %s",
                source_path,
                exc,
            )

            print(
                "\nERROR: The file could not be processed.\n"
                f"Reason: {exc}\n"
                "The application is still running. Try another file.\n"
            )


# INTERACTIVE LOOP

def print_banner() -> None:
    """Display startup information."""
    print("\n" + "=" * 100)
    print("SMARTNODE SUPPORT CALL TRIAGE PIPELINE")
    print("=" * 100)
    print("Audio input  : .mp3, .wav, .m4a, .aac, .flac, .ogg, .wma, .webm")
    print("Text input   : .txt, .md, .csv")
    print("Output       :", OUTPUT_CSV)
    print("Exit         : type 'exit' / 'quit' or press Ctrl+C")
    print("=" * 100)


def interactive_loop(app: CallTriageApplication) -> None:
    """Continuously ask for files until the user exits."""
    while True:
        try:
            raw_path = input(
                "\nEnter file path (or type 'exit'): "
            ).strip()

            if raw_path.lower() in {"exit", "quit"}:
                print("Exit requested. Closing application.")
                break

            if not raw_path:
                print("INPUT ERROR: File path cannot be empty.")
                continue

            try:
                source_path = validate_input_file(raw_path)
            except (FileNotFoundError, ValueError) as exc:
                print(f"INPUT ERROR: {exc}")
                continue

            app.process_file(source_path)

        except KeyboardInterrupt:
            print("\n\nCtrl+C received. Closing safely.")
            break

        except EOFError:
            print("\nInput stream closed. Exiting safely.")
            break

        except Exception as exc:
            logging.exception("Unexpected loop error: %s", exc)
            print(
                "Unexpected error occurred, but the application is still "
                f"running. Details: {exc}"
            )


# ENTRY POINT

def main() -> int:
    """Application entry point."""
    configure_logging()

    try:
        openai_key, hf_token = load_configuration()

        print_banner()

        app = CallTriageApplication(
            openai_key=openai_key,
            hf_token=hf_token,
        )

        interactive_loop(app)

        logging.info("Application closed successfully.")
        return 0

    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        return 0

    except Exception as exc:
        logging.exception("Fatal startup error: %s", exc)
        print(
            "\nFATAL ERROR: Application could not start.\n"
            f"Reason: {exc}\n"
            "Check your Python environment, dependencies, and .env settings."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
