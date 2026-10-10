import numpy as np, soundfile as sf, time
from kokoro import KPipeline

TEXT = ("I'm right here with you. You don't have to do anything, or be anything. "
        "Whatever you're feeling is welcome here, and I'm not going anywhere.")
VOICES = ["af_nicole", "af_heart", "af_bella", "af_sarah", "bf_emma", "bf_isabella", "am_michael", "bm_george"]
SPEED = 0.9

pipes = {}
for v in VOICES:
    lang = v[0]  # 'a' american, 'b' british
    pipe = pipes.setdefault(lang, KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M"))
    t = time.time()
    audio = np.concatenate([a for _, _, a in pipe(TEXT, voice=v, speed=SPEED)])
    sf.write(f"samples/{v}.wav", audio, 24000)
    print(f"{v}: {len(audio)/24000:.1f}s audio in {time.time()-t:.1f}s")
