import os
import io
import time
import json
import zipfile

import pandas as pd
from remotezip import RemoteZip
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FMA_DIR = os.path.join(DATA_DIR, "fma")
AUDIO_DIR = os.path.join(ROOT, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

GENRES = ["Pop", "Rock", "Hip-Hop", "Electronic", "Folk", "Experimental", "Instrumental", "International"]
PER_GENRE = 10
MAX_PER_ARTIST = 2
SMALL_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
OUT_PATH = os.path.join(DATA_DIR, "tracks.csv")


def load_metadata():
    tracks = pd.read_csv(os.path.join(FMA_DIR, "tracks.csv"), header=[0, 1], index_col=0)
    raw = pd.read_csv(os.path.join(FMA_DIR, "raw_tracks.csv"), usecols=[
        "track_id", "track_url", "track_file", "track_duration", "track_instrumental",
        "track_date_recorded", "license_title", "track_composer",
    ])
    raw = raw.set_index("track_id")
    return tracks, raw


def pick_tracks(tracks):
    subset = tracks[tracks[("set", "subset")] == "small"].copy()
    subset = subset[subset[("track", "genre_top")].isin(GENRES)]
    chosen = []
    artist_counts = {}
    for genre in GENRES:
        pool = subset[subset[("track", "genre_top")] == genre]
        pool = pool.sort_values(("track", "listens"), ascending=False)
        n = 0
        for track_id, row in pool.iterrows():
            artist = row[("artist", "name")]
            if artist_counts.get(artist, 0) >= MAX_PER_ARTIST:
                continue
            chosen.append((genre, int(track_id), row))
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            n += 1
            if n == PER_GENRE:
                break
    return chosen


def download_selected(track_ids):
    paths = {}
    with RemoteZip(SMALL_ZIP) as archive:
        names = set(archive.namelist())
        for i, track_id in enumerate(track_ids, 1):
            member = f"fma_small/{track_id // 1000:03d}/{track_id:06d}.mp3"
            if member not in names:
                raise FileNotFoundError(member)
            path = os.path.join(AUDIO_DIR, f"{track_id:06d}.mp3")
            if not os.path.exists(path) or os.path.getsize(path) < 1000:
                with archive.open(member) as source, open(path, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 256)
                        if not chunk:
                            break
                        target.write(chunk)
            paths[track_id] = path
            if i % 10 == 0:
                print("downloaded", i, "of", len(track_ids))
    return paths


def build():
    tracks, raw = load_metadata()
    chosen = pick_tracks(tracks)
    audio_paths = download_selected([track_id for _, track_id, _ in chosen])
    rows = []
    for genre, track_id, row in chosen:
        raw_row = raw.loc[track_id] if track_id in raw.index else pd.Series(dtype=object)
        rows.append({
            "track_uid": f"F{track_id:06d}",
            "track_id": track_id,
            "title": row[("track", "title")],
            "artist": row[("artist", "name")],
            "album": row[("album", "title")],
            "genre": genre,
            "genres_all": row[("track", "genres_all")],
            "release_date": row[("album", "date_released")],
            "listens": row[("track", "listens")],
            "favorites": row[("track", "favorites")],
            "duration_sec": row[("track", "duration")],
            "composer": raw_row.get("track_composer"),
            "instrumental_tag": raw_row.get("track_instrumental"),
            "license": raw_row.get("license_title"),
            "track_url": raw_row.get("track_url"),
            "track_file": raw_row.get("track_file"),
            "audio_path": audio_paths[track_id],
            "audio_member": f"fma_small/{track_id // 1000:03d}/{track_id:06d}.mp3",
        })
    df = pd.DataFrame(rows)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["era"] = pd.cut(
        df["year"], bins=[0, 1989, 1999, 2009, 2019, 2100],
        labels=["pre-1990", "1990s", "2000s", "2010s", "2020s"],
    ).astype(str)
    df.loc[df["era"] == "nan", "era"] = "unknown"
    df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df.groupby("genre").size().to_string())
    print(df[["genre", "title", "artist", "year"]].head(30).to_string(index=False))
