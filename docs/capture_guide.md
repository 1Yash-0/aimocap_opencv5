# AIMoCap - 3-phone test recording (developer take)

One continuous take, about 5-6 minutes. It is used for every pipeline stage (sync, calibration, 2D, 3D, hands, face, retarget), so please follow it exactly. Users will later do only steps 3-6 of the take; the extra parts are measuring tools for us.

## 1. What you need
- 3 smartphones (note make + model of each), charged > 50%, > 5 GB free each.
- 3 tripods or very stable stands (no handheld, no leaning on books that can slip).
- Tape measure, masking tape.
- Optional but very useful: a tablet or laptop that can show an image full-screen (calibration board file will be sent separately).
- A helper is nice but not required.

## 2. Room
- Free floor area about 3 x 3 m. Mark the centre with an X of tape.
- Bright, even light. No strong window behind the performer. Avoid flickering lights.
- Plain background if possible. No mirrors, TVs playing, or other people in view.
- Performer: fitted clothes that contrast with the background (no loose sleeves, no gloves, no hat). Bare hands. Hair away from the face if possible.

## 3. Phone placement
- Put the 3 phones in a triangle around the centre X, about 120 degrees apart (like three corners of a triangle), each 2.5-3.5 m from the centre.
- Height: about chest height (1.2-1.5 m), tilted slightly down.
- Landscape (horizontal).
- Check on every phone: the performer standing on the X with arms straight up is fully visible, feet included, with some free space around. Also check when standing 1 m off-centre in each direction.
- Once placed, DO NOT touch or move the phones until all three have stopped. If a phone moves, redo the whole take.

## 4. Phone settings (same on all 3)
Use the built-in camera app in video mode, or better a manual app (Open Camera on Android, Blackmagic Camera on iPhone/Android).
- Rear main camera, 1x (no ultra-wide, no tele, no zoom).
- 1080p, 60 fps on all three (if one phone cannot do 60, use 30 fps on ALL three).
- Turn OFF: video stabilization (EIS / "enhanced stabilization" / "action mode"), HDR, portrait/cinematic mode, beauty filters, auto low-light frame rate, auto zoom/framing.
- Lock focus and exposure: tap and hold on the performer at the centre X until it says AE/AF LOCK (or set manually: shutter about 1/120 s at 60 fps, fixed white balance).
- Airplane mode (no calls/notifications interrupting).

## 5. Measure and write down (before recording)
- Distance between each pair of phones (phone lens to phone lens): A-B, B-C, C-A.
- Height of each phone lens above the floor.
- Distance from each phone to the centre X.
- Performer height (without shoes if barefoot during recording).
- Phone make/model, app used, resolution, fps, which settings were turned off.
- Name the phones A, B, C and stick a small label on each.

## 6. The take (keep ALL phones recording the whole time)
Start recording on A, B, C (any order, does not matter), then walk to the X.

| # | Action | Duration | Notes |
|---|---|---|---|
| 1 | Clap 3 times, sharp and loud, hands clearly visible to all phones | 5 s | time sync |
| 2 | T-pose (arms straight out), stand still. Then A-pose (arms 45 deg down), still | 3 s + 3 s | bone lengths, rest pose |
| 3 | Calibration walk: walk slowly toward phone A and back to X, then toward B and back, then toward C and back. Then walk a full circle around the edge of the space. Swing arms naturally | 30 s | board-free calibration |
| 4 | Body: slow walk in a circle, raise arms forward/sideways/up, 3 squats, slow full 360 turn on the spot, sit on the floor and stand up | 60 s | general motion, feet on turns |
| 5 | Fast: arm swings, punches, kicks, 3-4 jumps, a fast turn | 30 s | fast motion |
| 6 | Hidden body parts: arms behind back, arms crossed, one hand behind the body while turning slowly | 20 s | occlusion handling |
| 7 | Hands (on the X, face each phone in turn for ~12 s): count 1 to 5 with fingers, fist then open hand, spread fingers, pinch, hand in front of face, clasp both hands | 40 s | finger tracking |
| 8 | Face: walk to about 1.5 m in front of ONE phone (write which). Neutral 2 s, blink a few times, open jaw wide, big smile, raise eyebrows, frown, look left/right/up/down | 20 s | facial expressions |
| 9 | Optional (only if tablet/laptop available): hold the screen showing the calibration board, walk slowly through the whole space tilting it in different directions so every phone sees it | 30 s | accuracy ruler, for developers only |
| 10 | Back to the X, clap 3 times again | 5 s | checks timing drift |

Then stop all three phones.

Write down approximately at what time (mm:ss on phone A) each step started.

## 7. Optional second short take
If time allows, do a second short take (steps 1-3 and 10 only, about 1 minute) WITHOUT moving the phones. It tells us whether calibration gives the same answer twice.

## 8. After recording
- Copy the ORIGINAL video files from each phone with a cable, Google Drive / iCloud "original quality", or AirDrop. Do NOT send through WhatsApp, Telegram or anything that recompresses video.
- Rename: `A_<phone model>.mp4`, `B_<phone model>.mp4`, `C_<phone model>.mp4` (keep the original files too).
- Put the videos and a `notes.txt` with everything from section 5 and the step times into one folder and share it.
- Take one photo of the room setup showing all three phones, if possible.

## Checklist
- [ ] Phones in a triangle, ~120 deg apart, 2.5-3.5 m from centre, chest height, landscape
- [ ] Same resolution and fps on all three, stabilization/HDR off, focus/exposure locked
- [ ] Full body visible on all phones with arms up
- [ ] Distances, heights, performer height written down
- [ ] Claps at start and end
- [ ] Phones never moved during the take
- [ ] Original files copied (no WhatsApp)
