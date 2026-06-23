# TODO

## Setup
- [x] Create repo with input MP4
- [x] Install PaddleOCR (most accurate open-source OCR engine)
- [x] Write frame analysis script
  - Dissect video into frames
  - Run PaddleOCR on each frame
  - Find bounding box for the word "World"
  - Draw a rectangle around it
  - Reassemble frames back into MP4
- [ ] Push to GitHub
- [ ] Set up hourly cron job

## Notes
- PaddleOCR chosen over Tesseract for superior text detection accuracy
- PaddleOCR uses PP-OCRv4 model — state-of-the-art for scene text
- Script extracts every frame at source FPS, processes, reassembles at same FPS