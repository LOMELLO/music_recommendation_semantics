import os
import json
import time

import numpy as np
import pandas as pd
import torch
import librosa
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    ClapModel,
    ClapProcessor,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CLAP_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLAP_MODEL = "laion/larger_clap_music"
MOOD_MODEL = "StanislavKo28/music_moods_classification"
SR = 48000
CROP_SECONDS = 10

MOOD_GROUPS = {
    "angry": "aggressive",
    "dark": "dark",
    "energetic": "energetic",
    "epic": "epic",
    "euphoric": "happy",
    "happy": "happy",
    "mysterious": "dark",
    "relaxing": "calm",
    "romantic": "romantic",
    "sad": "sad",
    "scary": "dark",
    "glamorous": "happy",
    "uplifting": "happy",
    "sentimental": "nostalgic",
}
GROUP_ORDER = ["happy", "energetic", "calm", "romantic", "nostalgic", "sad", "dark", "aggressive", "epic"]
CONTEXTS = ["workout", "party", "study", "relaxation", "commute", "romantic evening", "sleep"]
ATMOSPHERES = ["uplifting", "dreamy", "dark", "warm", "melancholic", "aggressive", "intimate", "cinematic"]


def load_clap():
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = ClapModel.from_pretrained(CLAP_MODEL).eval().to(CLAP_DEVICE)
    return processor, model


def load_mood_model():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MOOD_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(MOOD_MODEL).eval()
    return feature_extractor, model


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < CROP_SECONDS * SR:
        y = np.pad(y, (0, CROP_SECONDS * SR - len(y)))
    return y


def make_crops(y):
    width = CROP_SECONDS * SR
    starts = [0, max(0, (len(y) - width) // 2), max(0, len(y) - width)]
    return [y[s:s + width].astype(np.float32) for s in starts]


def clap_embedding(processor, model, crops):
    inputs = processor(audio=crops, sampling_rate=SR, return_tensors="pt")
    inputs = {k: v.to(CLAP_DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.get_audio_features(**inputs).pooler_output
    output = output.mean(dim=0)
    return torch.nn.functional.normalize(output, dim=0).detach().cpu().numpy()


def mood_probabilities(feature_extractor, model, path):
    y, _ = librosa.load(path, sr=feature_extractor.sampling_rate, mono=True, duration=30)
    inputs = feature_extractor(y, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    label_map = {int(k): v for k, v in model.config.id2label.items()}
    labels = [label_map[i] for i in range(len(probs))]
    return labels, probs


def group_mood(labels, probs):
    grouped = {g: 0.0 for g in GROUP_ORDER}
    for label, prob in zip(labels, probs):
        grouped[MOOD_GROUPS[label]] += float(prob)
    return grouped


def dsp_features(path):
    y, _ = librosa.load(path, sr=22050, mono=True)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=22050)[0])[0])
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=22050)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=22050)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=22050)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=20).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=22050).mean(axis=1)
    values = {
        "tempo_bpm": tempo,
        "rms_energy": float(rms.mean()),
        "rms_std": float(rms.std()),
        "spectral_centroid": float(centroid.mean()),
        "spectral_bandwidth": float(bandwidth.mean()),
        "zero_crossing_rate": float(zcr.mean()),
        "spectral_rolloff": float(rolloff.mean()),
        "clip_duration_sec": float(len(y) / 22050),
    }
    vector = np.concatenate([[values[k] for k in [
        "tempo_bpm", "rms_energy", "rms_std", "spectral_centroid",
        "spectral_bandwidth", "zero_crossing_rate", "spectral_rolloff",
    ]], mfcc, chroma]).astype(np.float32)
    names = [
        "tempo_bpm", "rms_energy", "rms_std", "spectral_centroid",
        "spectral_bandwidth", "zero_crossing_rate", "spectral_rolloff",
    ] + [f"mfcc_{i}" for i in range(1, 21)] + [f"chroma_{i}" for i in range(1, 13)]
    return values, vector, names


def context_scores(mood, energy_level, tempo_category, instrumental):
    fast = tempo_category == "fast"
    medium = tempo_category == "medium"
    slow = tempo_category == "slow"
    high = energy_level == "high"
    low = energy_level == "low"
    medium_energy = energy_level == "medium"
    instr = instrumental == "instrumental"
    return {
        "workout": 3 * ((mood in {"dark", "aggressive", "epic"}) and high) + 2 * (mood == "happy" and high and instr) + 2 * fast + high,
        "party": 2 * (mood == "happy" and high) + (mood == "happy" and medium_energy) + (mood == "romantic" and high) + (not instr),
        "study": 3 * (instr and (low or medium_energy)) + 2 * (mood == "dark" and (low or medium_energy)) + (mood in {"calm", "nostalgic"}),
        "relaxation": 3 * (mood == "happy" and low) + 2 * (mood in {"calm", "romantic"} and (low or medium_energy)) + 2 * (mood == "sad" and medium_energy) + low,
        "commute": 3 * (mood == "happy" and medium_energy) + 2 * (mood == "happy" and high and medium) + (mood == "happy" and low),
        "romantic evening": 5 * (mood == "romantic" and (low or medium_energy)) + 2 * (mood == "nostalgic"),
        "sleep": 3 * (mood == "sad" and low) + 2 * (mood == "calm" and slow) + (mood == "dark" and low and slow),
    }


def atmosphere_scores(mood, fine_mood, energy_level, tempo_category, instrumental, spectral_centroid):
    fast = tempo_category == "fast"
    medium = tempo_category == "medium"
    slow = tempo_category == "slow"
    high = energy_level == "high"
    low = energy_level == "low"
    medium_energy = energy_level == "medium"
    instr = instrumental == "instrumental"
    warm_timbre = spectral_centroid < 1800
    return {
        "uplifting": 3 * (mood == "happy") + high + fast,
        "warm": 2 * (mood == "happy") + low + slow + warm_timbre,
        "dreamy": 3 * (mood in {"calm", "nostalgic"}) + 2 * low + 2 * slow + instr,
        "melancholic": 5 * (mood == "sad") + 2 * (mood == "nostalgic") + low + slow,
        "aggressive": 5 * (mood == "aggressive") + 2 * (fine_mood in {"angry", "scary"}) + 2 * high + fast,
        "intimate": 3 * (mood in {"romantic", "calm"}) + 2 * low + slow + instr,
        "cinematic": 4 * (mood == "epic") + 3 * (fine_mood in {"epic", "scary"}) + 2 * instr + (high or medium_energy),
        "dark": 4 * (mood == "dark") + 2 * (fine_mood in {"dark", "mysterious", "scary"}) + (low or medium_energy),
    }


def choose(scores):
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(max(v, 0.01) for _, v in ordered)
    label, top = ordered[0]
    second = ordered[1][1]
    return label, float(top / total), float((top - second) / total)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "tracks.csv"))
    df["instrumental"] = df["instrumental_tag"].map({1: "instrumental", 0: "vocal"})

    clap_processor, clap_model = load_clap()
    mood_feature_extractor, mood_model = load_mood_model()

    embeddings = []
    dsp_vectors = []
    dsp_names = None
    mood_prob_matrix = []
    mood_group_matrix = []
    context_score_matrix = []
    atmosphere_score_matrix = []
    feature_records = []
    mood_labels = None

    start = time.time()
    for i, row in df.iterrows():
        y = load_audio(row["audio_path"])
        emb = clap_embedding(clap_processor, clap_model, make_crops(y))
        embeddings.append(emb)

        values, vector, names = dsp_features(row["audio_path"])
        dsp_names = names
        dsp_vectors.append(vector)

        labels, probs = mood_probabilities(mood_feature_extractor, mood_model, row["audio_path"])
        mood_labels = labels
        grouped = group_mood(labels, probs)
        fine_mood = labels[int(np.argmax(probs))]
        mood = max(grouped, key=grouped.get)
        mood_conf = grouped[mood]
        mood_margin = mood_conf - sorted(grouped.values(), reverse=True)[1]
        mood_prob_matrix.append(probs)
        mood_group_matrix.append([grouped[g] for g in GROUP_ORDER])

        feature_records.append({
            "mood": mood,
            "mood_fine": fine_mood,
            "mood_confidence": mood_conf,
            "mood_margin": mood_margin,
            "mood_fine_confidence": float(probs.max()),
            **values,
        })

        record = feature_records[-1]
        record["_dsp_values"] = values
        record["_mood_group"] = grouped
        record["_index"] = i

        print(i + 1, "of", len(df), round(time.time() - start, 1), "sec")

    df = pd.concat([df, pd.DataFrame([{
        "mood": r["mood"], "mood_fine": r["mood_fine"],
        "mood_confidence": r["mood_confidence"], "mood_margin": r["mood_margin"],
        "mood_fine_confidence": r["mood_fine_confidence"],
        **r["_dsp_values"],
    } for r in feature_records], index=df.index)], axis=1)

    df["energy_level"] = pd.cut(
        df["rms_energy"], bins=[-np.inf, df["rms_energy"].quantile(0.33), df["rms_energy"].quantile(0.66), np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    df["tempo_category"] = pd.cut(
        df["tempo_bpm"], bins=[0, 90, 120, 300], labels=["slow", "medium", "fast"]
    ).astype(str)

    contexts, atmosphere = [], []
    for i, row in df.iterrows():
        cs = context_scores(row["mood"], row["energy_level"], row["tempo_category"], row["instrumental"])
        ats = atmosphere_scores(
            row["mood"], row["mood_fine"], row["energy_level"], row["tempo_category"],
            row["instrumental"], row["spectral_centroid"],
        )
        context, context_conf, context_margin = choose(cs)
        atmos, atmos_conf, atmos_margin = choose(ats)
        contexts.append((context, context_conf, context_margin, [cs[c] for c in CONTEXTS]))
        atmosphere.append((atmos, atmos_conf, atmos_margin, [ats[a] for a in ATMOSPHERES]))

    df["context"] = [x[0] for x in contexts]
    df["context_confidence"] = [x[1] for x in contexts]
    df["context_margin"] = [x[2] for x in contexts]
    df["atmosphere"] = [x[0] for x in atmosphere]
    df["atmosphere_confidence"] = [x[1] for x in atmosphere]
    df["atmosphere_margin"] = [x[2] for x in atmosphere]
    context_score_matrix = [x[3] for x in contexts]
    atmosphere_score_matrix = [x[3] for x in atmosphere]

    embeddings = np.stack(embeddings).astype(np.float32)
    np.save(os.path.join(DATA_DIR, "clap_embeddings.npy"), embeddings)
    np.savez(
        os.path.join(DATA_DIR, "semantic_probabilities.npz"),
        mood_fine=np.stack(mood_prob_matrix).astype(np.float32),
        mood_group=np.stack(mood_group_matrix).astype(np.float32),
        context=np.stack(context_score_matrix).astype(np.float32),
        atmosphere=np.stack(atmosphere_score_matrix).astype(np.float32),
        mood_fine_labels=np.array(mood_labels),
        mood_group_labels=np.array(GROUP_ORDER),
        contexts=np.array(CONTEXTS),
        atmospheres=np.array(ATMOSPHERES),
    )
    np.savez(
        os.path.join(DATA_DIR, "dsp_features.npz"),
        features=np.stack(dsp_vectors).astype(np.float32), names=np.array(dsp_names),
    )
    df.to_csv(os.path.join(DATA_DIR, "tracks_enriched.csv"), index=False)
    with open(os.path.join(DATA_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "clap_model": CLAP_MODEL, "mood_model": MOOD_MODEL,
            "sample_rate": SR, "crop_seconds": CROP_SECONDS,
            "mood_groups": MOOD_GROUPS, "contexts": CONTEXTS, "atmospheres": ATMOSPHERES,
        }, f, ensure_ascii=False, indent=2)
    print(df[["title", "artist", "genre", "mood", "mood_fine", "context", "atmosphere", "tempo_bpm", "rms_energy"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
