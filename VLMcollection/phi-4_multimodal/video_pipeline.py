import argparse
import sys

import cv2
import soundfile as sf
import torch
from PIL import Image

from inference import build_prompt
from model_loader import load_model


def sample_frames(video_path: str, frame_interval: int = 30, max_frames: int = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.1f}s")
    print(f"  Sampling every {frame_interval} frame(s)")

    frames = []
    timestamps = []
    frame_idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)
            timestamps.append(frame_idx / fps)
            sampled += 1
            if max_frames and sampled >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"  Sampled {len(frames)} frame(s)\n")
    return frames, timestamps


def process_video(
    model,
    processor,
    video_path: str,
    text: str = None,
    audio_path: str = None,
    frame_interval: int = 30,
    max_frames: int = None,
    max_new_tokens: int = 500,
    do_sample: bool = False,
    temperature: float = None,
    top_p: float = None,
):
    frames, timestamps = sample_frames(video_path, frame_interval, max_frames)

    if not frames:
        print("No frames sampled from video.")
        return

    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        print(f"--- Frame {i+1}/{len(frames)} @ t={ts:.2f}s ---")

        prompt = build_prompt(text=text, has_image=True, has_audio=audio_path is not None)
        print(f"Prompt: {prompt}")

        audios = None
        if audio_path:
            audio_data, samplerate = sf.read(audio_path)
            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1)
            audios = [(audio_data, samplerate)]

        inputs = processor(
            text=prompt,
            images=frame,
            audios=audios,
            return_tensors="pt",
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_logits_to_keep": 0,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generate_ids = model.generate(**inputs, **gen_kwargs)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        print(f"Response: {response}\n")


def process_webcam(
    model,
    processor,
    text: str = None,
    camera_id: int = 0,
    frame_interval: int = 30,
    max_new_tokens: int = 500,
    do_sample: bool = False,
    temperature: float = None,
    top_p: float = None,
):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera {camera_id}")

    print(f"Webcam ({camera_id}) started. Press Ctrl+C to stop.\n")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = build_prompt(text=text, has_image=True)
                print(f"--- Frame @ t≈{frame_idx/30:.1f}s ---")
                print(f"Prompt: {prompt}")

                inputs = processor(
                    text=prompt,
                    images=pil_img,
                    return_tensors="pt",
                ).to(model.device)

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "num_logits_to_keep": 0,
                }
                if do_sample:
                    gen_kwargs["do_sample"] = True
                if temperature is not None:
                    gen_kwargs["temperature"] = temperature
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p

                with torch.no_grad():
                    generate_ids = model.generate(**inputs, **gen_kwargs)
                    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
                    response = processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                print(f"Response: {response}\n")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Phi-4-multimodal-instruct video pipeline"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device ID")
    parser.add_argument("--text", type=str, default="Describe what you see in this image.", help="Text prompt")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (optional, combined with video)")
    parser.add_argument("--frame-interval", type=int, default=30, help="Sample every Nth frame (default: 30)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    args = parser.parse_args()

    if not args.video and args.webcam is None:
        print("Error: Specify --video <path> or --webcam <device_id>", file=sys.stderr)
        sys.exit(1)

    model, processor = load_model(
        cache_dir=args.cache_dir or None,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
    )

    if args.video:
        process_video(
            model=model,
            processor=processor,
            video_path=args.video,
            text=args.text,
            audio_path=args.audio,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sampling,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.webcam is not None:
        process_webcam(
            model=model,
            processor=processor,
            text=args.text,
            camera_id=args.webcam,
            frame_interval=args.frame_interval,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sampling,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
