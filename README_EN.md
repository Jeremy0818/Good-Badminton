# Good-Badminton: AI Badminton Hawk-Eye System 🏸

## Related Projects

Good-Badminton, Good-Tennis, and Good-Pickleball are part of the same family of computer-vision sports video analysis projects. They share the same core ideas: player detection, ball trajectory tracking, court coordinate mapping, movement statistics, and visualized outputs. Each project adapts the court model, ball target, and sport-specific rules to a different sport.

| Project | Sport | Stars |
| --- | --- | --- |
| [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) | Badminton video analysis | [![Good-Badminton stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/stargazers) |
| [Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis) | Tennis video analysis | [![Good-Tennis stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Tennis?style=social)](https://github.com/yo-WASSUP/Good-Tennis/stargazers) |
| [Good-Pickleball](https://github.com/yo-WASSUP/Good-Pickleball) | Pickleball video analysis | [![Good-Pickleball stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Pickleball?style=social)](https://github.com/yo-WASSUP/Good-Pickleball/stargazers) |

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yo-WASSUP/Good-Badminton?style=social)](https://github.com/yo-WASSUP/Good-Badminton/network/members)
[![GitHub license](https://img.shields.io/github/license/yo-WASSUP/Good-Badminton)](https://github.com/yo-WASSUP/Good-Badminton/blob/main/LICENSE)
[![RedNote](https://img.shields.io/badge/RedNote-ff2442)](https://www.xiaohongshu.com/explore/6a37b1d20000000011016229?xsec_token=ABod3wXBTiDppp6W2Ou0QHlu2eotUkeu27-ha64nFRR74=&xsec_source=pc_user)

**A computer-vision toolkit for badminton match video analysis**

[中文](README.md) | [English](README_EN.md)

</div>

## 🎬 Preview

![Good-Badminton analysis preview](assets/demo_en.gif)


## 🆕 Changelog

- **2026-06-27**: Improved automatic court-line detection.
- **2026-06-24**: Added Gradio WebUI, - From KangweiLIAO PR.
- **2026-06-23**: Added automatic court boundary detection.
- **2026-06-20**: Initial open-source release.
- **2026-06-17**: Project documentation cleanup.
- **Current version**: Supports player pose detection, shuttlecock detection, court coordinate mapping, trajectory statistics, heatmaps, scatter plots, and annotated video output.
- **Experimental features**: Hit-point analysis and stroke statistics are still under active iteration and are mainly intended for research and secondary development.

## 🔮 Roadmap

- [x] Frame-by-frame badminton match video analysis
- [x] RTMPose / RTMO / YOLO Pose model support
- [x] YOLO shuttlecock detection model integration
- [x] Manual court annotation and court coordinate mapping
- [x] Player movement trajectory, speed, distance, and rally statistics
- [x] Chinese / English visualization text
- [x] Heatmap, scatter plot, and detection data export
- [ ] More stable hit-point recognition
- [ ] More accurate shuttlecock detection model
- [ ] More complete stroke statistics
- [x] Automatic court keypoint detection
- [x] Gradio WebUI (browser-based, no CLI required)
- [ ] Batch video analysis workflow

---

## ✨ Features

- **Player pose detection** - Supports RTMPose, RTMO, and Ultralytics YOLO Pose for human keypoint and skeleton detection.
- **Shuttlecock detection** - Uses a YOLO model to detect shuttlecock positions and draw trajectories in the output video.
- **Court coordinate mapping** - Manually annotates court keypoints and maps image coordinates to standard badminton court coordinates.
- **Automatic court detection** - Matches visible white/yellow court lines against a standard badminton court model, with manual corner correction available in the WebUI.
- **Player position tracking** - Tracks upper-court and lower-court players separately and records movement trajectories.
- **Rally detection** - Detects rally start/end states from continuous court-view matching and records rally IDs in overlays and detection data.
- **Motion statistics** - Computes movement distance, current speed, maximum speed, and rally counts.
- **Visual output** - Generates annotated videos with skeletons, trajectories, statistics, and court trajectory overlays.
- **Position charts** - Automatically generates player position heatmaps and scatter plots.
- **Chinese / English display** - Switch visualization text with `--language zh/en`.
- **WebUI** - Browser-based interface built with Gradio. Upload videos, detect the court, configure parameters, and view results without touching the command line.
- **Local execution** - Videos, models, and analysis outputs stay on your local machine.

## Requirements

- Python 3.8+
- FFmpeg available in system `PATH`
- Shuttlecock YOLO detection weight, downloaded from [GitHub Releases](https://github.com/yo-WASSUP/Good-Badminton/releases/latest)

## Performance Requirements and Reference Speed

Recommended setup:

- GPU with 6GB+ VRAM. More VRAM helps with higher-resolution videos and larger pose models.
- 16GB+ system RAM.
- SSD storage for output videos, `detections.jsonl`, and visualization images.
- CPU execution is supported, but pose detection and shuttlecock detection will be much slower. It is best suited for short clips or feature checks.

Actual speed depends on the GPU, video resolution, pose model, preview display, and audio export settings.

For a 720p video with `--pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt` and `weights/yolo11s-ball.pt`, GPU timing logs are typically close to:

```text
pose 0.02s, shuttlecock 0.02s, shuttle draw 0.00s, players draw 0.01s, court draw 0.00s
```

Use `--performance-stats` to print a compact timing summary about every 5 seconds and identify whether the bottleneck is pose inference, shuttlecock detection, or drawing.

## 🚀 Installation

The default dependencies use CPU PyTorch and ONNX Runtime.

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Acceleration (Windows / NVIDIA)

Prerequisites:

- NVIDIA driver installed, and `nvidia-smi` works correctly.
- CUDA 12.1 PyTorch wheels are recommended.

PowerShell:

```bash
.\.venv\Scripts\activate

pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

Verify GPU availability:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected output includes:

```text
cuda: True
CUDAExecutionProvider
```

> Note: after installing GPU ONNX Runtime, `pip check` may report `rtmlib requires onnxruntime, which is not installed`. If provider verification shows `CUDAExecutionProvider`, do not reinstall CPU `onnxruntime`, because it may overwrite the GPU package.

Switch back to CPU dependencies:

```bash
pip install --force-reinstall -r requirements.txt
```


### WebUI Installation (Optional)

The WebUI is built with Gradio and requires an additional dependency:

```bash
pip install -r requirements-webui.txt
```

Launch the WebUI:

```bash
python -m webui.app
```

Open the URL printed in the terminal (default `http://127.0.0.1:7860`) and:

1. Upload a match video and a court template image.
2. Click "Detect Court" to auto-detect court boundaries. To correct, click 4 corners on the image then click "Apply Manual Corners".
3. Adjust analysis settings (pose model, language, visualization options, etc.).
4. Click "Run Analysis" and wait for the progress bar. Results include the annotated video, heatmaps/scatter plots, and detection data.

| Court Detection and Settings | Analysis Results |
| --- | --- |
| ![WebUI court detection screen](assets/webui0-en.png) | ![WebUI results screen](assets/webui1-en.png) |

The WebUI is optional — the CLI workflow is unchanged.

## 📝 Usage

### First Run Workflow (CLI)

1. Prepare the input video and shuttlecock detection weight.
2. Run the basic command:

```bash
python main.py --video-path videos/demo.mp4
```

3. If `--template-path` is not provided, the program opens a file picker for a court template image. Usually, choose a stable frame with clear court lines.
4. The program first tries to detect the court boundary automatically and saves `outputs/<video_name>/auto_court_preview.png`. Press Enter/Y in the preview window to accept it, or press M/R/Esc to switch to manual four-corner annotation.
5. If manual annotation is used, follow the prompt at the top of the image and click the four court corners in order: top-left, top-right, bottom-right, bottom-left.

![Court annotation example](assets/label_court_example.png)

6. After the four points are selected, the window shows a green court box and a blue pose-detection ROI. The ROI is generated automatically from the court area.
7. The annotation is saved to `outputs/<video_name>/court_annotations.txt`. Re-running with the same output directory reuses this file.
8. After analysis finishes, check `outputs/<video_name>/detect_<video_name>.mp4`, `detections.jsonl`, and `position_visualizations/`.

Why four court points are required:

- The four corners establish the mapping from image coordinates to standard badminton court coordinates.
- Player filtering mainly depends on court coordinates, which helps remove spectators, referees, and people outside the court.
- Upper/lower court player assignment, movement distance, speed, rally statistics, heatmaps, and scatter plots all depend on this mapping.
- Rally detection uses court template matching: consecutive court-view frames start a rally, and consecutive non-court-view frames end it.
- The pose ROI only reduces the inference area and improves speed. It is automatically expanded from the court area.
- Shuttlecock detection still runs on the full frame, with basic filtering based on the horizontal court range plus padding.

If the video angle, crop, or template image changes, delete the corresponding `court_annotations.txt` and annotate the four points again.


### Pose Model Selection

```bash
# Default: two-stage RTMPose balanced
python main.py --video-path videos/demo.mp4 --pose-family rtmpose --pose-mode balanced

# Lighter one-stage RTMO
python main.py --video-path videos/demo.mp4 --pose-family rtmo --pose-mode lightweight

# Use Ultralytics YOLO Pose
python main.py --video-path videos/demo.mp4 --pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt
```

RTMPose / RTMO modes:

- `lightweight`: prioritizes speed.
- `balanced`: default tradeoff between speed and quality.
- `performance`: larger model, slower, usually better for detection quality.

### Common Arguments

```text
--video-path                 Input video path, required
--output-dir                 Output directory, default outputs/<video_name>
--ball-model                 YOLO shuttlecock detection model path, default weights/yolo11s-ball.pt
--pose-family                Pose model family: rtmpose, rtmo, or yolo-pose
--pose-mode                  RTMPose / RTMO mode: lightweight, balanced, performance
--yolo-pose-model            YOLO pose model path or model name, default yolo11n-pose.pt
--template-path              Court template image path; opens a file picker if omitted
--pose-roi true|false                Show pose-detection ROI box, default true
--display true|false                 Show OpenCV preview window, default true
--skeletons true|false               Show human skeletons, default true
--player-trajectories true|false     Show player trajectories, default true
--court-trajectory true|false        Show court trajectory overlay, default true
--shuttlecock-trajectory true|false  Show shuttlecock trajectory, default true
--player-stats true|false            Show player statistics, default true
--performance-stats                  Print performance timings
--save-images                        Save processed frame images
--visualize-positions true|false     Generate heatmaps and scatter plots, default true
--audio true|false                   Keep original video audio, default true
--language {zh,en}                   Visualization language
--line-calls true|false              Call each rally's landing in or out (experimental), default false
--match-type singles|doubles         Court boundary used for line calls, default singles
```

### Line Calls (Experimental)

`--line-calls true` finds where the last shot of each rally landed and calls it in or out. Rallies are split from shuttle motion, so it works on fixed-camera recordings with no camera cuts.

- A landing is where the shuttle first touches the floor. It often bounces or skids on 0.3 m or more before it stops, so the touch-down is worked out from the moment its fast flight ends, between video frames. A shuttle that slides on fast is placed where its speed across the court drops from flight speed to sliding speed. A shuttle that stops next to a player's hand is treated as caught, not landed.
- A shuttle on the line is in. Each call reports its distance to the nearest boundary line in cm.
- Calls are flagged for review when the shuttle was lost before it settled, was caught, landed within 15 cm of a line or 30 cm of the net, or seemed to land more than 3 m outside the court (no shot lands that far out, so it was still in the air).
- Outputs: `line_calls.json`, `line_calls_map.png` (landing points on a court diagram), and `line_calls_<video_name>.mp4` (landing point and call drawn on the video).

Re-run line calls on an existing output directory without re-processing the video:

```bash
python -m badminton_analysis.analysis.line_calls outputs/<video_name> --match-type singles --video
```

Single-camera calls are reliable for clear landings. Shots within a few cm of a line, or landings at 25-30 fps where the shuttle is blurred, still need a human check.

For more accurate landings, track the shuttle a second time with [TrackNetV3](https://github.com/qaz812345/TrackNetV3), which looks at 8 frames at once and keeps hold of the small, blurred shuttle near the floor better than the single-frame detector. Download TrackNetV3's [checkpoints](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing), unzip them and put `TrackNet_best.pt` in `weights/`, then run:

```bash
python -m badminton_analysis.detection.tracknet outputs/<video_name>
```

It saves `tracknet.csv` next to `detections.jsonl`, and line calls, scoring and both reviews use its shuttle positions from then on (delete the file to go back to the detector's). On an Apple M3 Pro it takes about 22 ms per analysed frame of 1080p video, faster than the video plays. On four broadcast test matches it raised correct line calls from 106 of 142 to 121 of 146 and cut the median distance from the labelled landing spot from 30 to 22 cm.

### Scoring with Review

Automatic scoring gets roughly 70% of rally winners right on broadcast test matches, so the score is built with a quick review: each rally's ending is replayed with the automatic call drawn on it, and one key confirms or corrects it.

```bash
python -m badminton_analysis.analysis.review outputs/<video_name> --names "Lee,Tan" --player1-side lower
```

| Key | Action |
| --- | --- |
| Enter / Space | Accept the automatic winner |
| `n` / `f` | Near / far side won the rally |
| `x` | Not a rally (warm-up, replay, shuttle knocked back) |
| `r` / `b` | Replay the clip / go back one rally |
| `q` / Esc | Save and quit; run again to continue where you left off |

- `--player1-side` is the side player 1 starts on (`lower` = near the camera). Ends change automatically after each game and at 11 in game 3.
- `--start-score 12-6` and `--start-games 1-0` set the score when the video starts part-way through a match.
- `--skip-high` accepts high-confidence calls without showing them.
- Decisions are saved to `review.json` after every key; the score is in `review_report.json`.

The same review runs in the browser, with the landing marked on the video, a court map of every landing, and the rally list with the running score:

```bash
python -m badminton_analysis.analysis.review_web outputs/<video_name> --names "Lee,Tan" --player1-side lower
# then open http://127.0.0.1:8765
```

It uses the same keys plus ← / → to move between rallies, and saves to the same files, so you can switch between the two.

Under the video, the touch-down is replayed zoomed in at quarter speed (frame by frame if you like) with the court lines drawn where the system places them, so you can see they sit on the painted lines, and the shuttle's tracked positions. Next to it, a 3D close-up shows the mark, its error ring and how sure the call is, for example "OUT by 5 cm · ±38 cm · 55% sure · too close to call". Drag to turn it.

### Filming with a Phone (Experimental)

The phone films and this computer records and decodes, so the phone only needs a browser (iPhone Safari or Android Chrome):

```bash
python -m badminton_analysis.live.capture
# then open http://127.0.0.1:8080 on this computer
```

1. The page on the computer shows an address like `https://<computer>.local:8443/?k=…`. Open it on the phone. Phone and computer must be on the same Wi-Fi; at a hall, turn on the iPhone's Personal Hotspot and join the computer to it (the video stays between the two devices and uses no mobile data).
2. The computer makes its own certificate, so Safari warns once that the connection is not private: Show Details › visit this website. Allow the camera and microphone (hit sounds are kept for later analysis).
3. Put the phone sideways on a stand, pick the camera and quality, and press Start. Keep Safari open with the screen on (Auto-Lock › Never) and the phone on a power bank.
4. The computer page shows the latest picture, how far decoding trails the phone, and missing pieces. Recordings are saved to `recordings/<date-time>/recording.mp4`, ready for `python main.py --video-path …`.

Placement: beside the court in line with the net, as high as possible, with the whole court in the picture (usually the 0.5× camera, if the phone's browser offers it). From there every hit reverses the shuttle's direction across the picture, which makes hits and hitters easy to find. High behind a baseline also works.

### Tracking and Calling While Filming (Experimental)

The first part of live review tracks the shuttle and players as frames arrive, with TrackNet for the shuttle and a pose a few times a second, and makes each rally's line call as soon as nothing still to come can change it. Until it is joined to the phone, try it by replaying an analysed video at its own speed:

```bash
python -m badminton_analysis.live.tracker outputs/<video_name> --start 600 --seconds 120
```

It prints each call as it is made, writes `live_detections.jsonl` and `live_line_calls.json` next to `detections.jsonl`, and reports the time each step takes. On an Apple M3 Pro, two minutes of a broadcast match took 15 ms a frame of the 33 ms available at 30 fps, and each frame's record was ready 0.3 s after the frame.

A call waits until the shuttle has been followed for 2.5 s after its last flight, so it comes about 3 s after the landing. On four ShuttleSet matches the live calls were the same as those of the full analysis for all 280 rallies. Two came more than 4 s after the landing: in both the shuttle was lost, and it was flying again straight after.

With the phone: once it is in place, press **Mark the court** on this computer's page. It suggests the court's corners; drag the four dots onto them until the drawn lines sit on the court's lines. From then on the page lists each rally's call as it is made. To try this without a phone, send a saved video in as if a phone were filming it:

```bash
python -m badminton_analysis.live.capture --replay videos/<video>.mp4 --start 638 --seconds 60
```

Keep the phone still once the court is marked: a shift of a few pixels moves landings near the far baseline by 10 cm or more. If the picture moves 2 px or more, the page says so; mark the court again.

Not fast enough yet for a phone: on an Apple M3 Pro, tracking every frame of a 1080p video still takes about 32 ms of the 33 ms a frame lasts at 30 fps, so tracking can fall several seconds behind. A broadcast catches up during replays and close-ups. `python -m badminton_analysis.live.tracker ... --every-frame` measures this load.

## 📊 Outputs

Default output directory: `outputs/<video_name>/`.

- `metadata.json`: metadata for video, models, court annotation, and output files.
- `detections.jsonl`: per-frame detection records, including rally ID, players, hands, court coordinates, speed, and shuttlecock coordinates.
- `detect_<video_name>.mp4`: annotated output video with skeletons, trajectories, statistics, and rally IDs.
- `court_annotations.txt`: cached court annotation coordinates.
- `position_visualizations/heatmaps/`: player position heatmaps.
- `position_visualizations/scatter_plots/`: player position scatter plots.

### Position Visualization Examples

| Heatmap | Scatter Plot |
| --- | --- |
| ![Player position heatmap example](assets/match_heatmap_en.png) | ![Player position scatter plot example](assets/match_scatter_en.png) |

## 🧩 Project Structure

```text
main.py              # CLI entry and argument parsing; keeps python main.py ... usage
badminton_analysis/
├── system.py        # Main video analysis pipeline: BadmintonAnalysisSystem
├── analysis/        # Line calls, scoring and rally review
├── court/           # Court annotation and coordinate mapping
├── data/            # JSON / JSONL output
├── detection/       # Shuttlecock detection and pose detection
├── live/            # Phone camera capture: record and decode on this computer
├── media/           # Video/audio processing
├── tracking/        # Player tracking
└── visualization/   # Video overlays, statistics charts, and position plots
webui/
├── app.py           # Gradio WebUI interface and launch entry point
└── pipeline.py      # WebUI analysis pipeline orchestration
```

## Acknowledgements

Thanks to the RTMPose, RTMO, and OpenMMLab ecosystem for the pose-estimation foundations, and to [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) for the lightweight pose-estimation runtime.

Thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO object-detection algorithms and tooling.

Thanks to [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) for organizing and releasing badminton datasets, which provided important references for shuttlecock detection and trajectory analysis in this project.

Thanks to [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) for the shuttle-tracking model and checkpoints used for more accurate landings.

## 📄 License

Project code and `weights/yolo11s-ball.pt` are licensed under Apache License 2.0. RTMPose / RTMO / YOLOX ONNX weights provided in Releases come from the OpenMMLab / RTMPose ecosystem, are used under their upstream Apache License 2.0, and retain their original attribution. TrackNetV3's checkpoints, downloaded separately, and the network in `badminton_analysis/detection/tracknet.py`, which follows its code, are under TrackNetV3's MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yo-WASSUP/Good-Badminton&type=Date)](https://www.star-history.com/#yo-WASSUP/Good-Badminton&Date)

