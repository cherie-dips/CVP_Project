"""Download more Objaverse meshes, so the selector can be trained and tested on more than 100 objects.

The objects come from GraspGen's Objaverse subset (data/objaverse/scales.json, 8,515 objects with
real-world scale factors), so every one has a known size in metres. The 100 objects the first
fine-tune used are already in data/objaverse/meshes; this adds N more, chosen at random with a fixed
seed, and records their names from the Objaverse metadata.

    /opt/anaconda3/envs/graspgenx/bin/python stage1_gripper/scripts/download_objaverse.py [N]

Writes data/objaverse/meshes/<uid>.glb and adds the new names to data/objaverse/annotations.json.
"""
import concurrent.futures as cf, gzip, json, os, pathlib, random, shutil, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MESHES = ROOT/"data"/"objaverse"/"meshes"
CACHE = pathlib.Path.home()/".objaverse"/"hf-objaverse-v1"       # the objaverse package's cache
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def fetch(rel, dest):
    """Copy from the objaverse cache if it is there, else download."""
    if (CACHE/rel).exists():
        shutil.copy(CACHE/rel, dest)
        return
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(HF + rel, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def main():
    if not (CACHE/"object-paths.json.gz").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        fetch("object-paths.json.gz", CACHE/"object-paths.json.gz")
    paths = json.load(gzip.open(CACHE/"object-paths.json.gz"))
    scales = json.loads((ROOT/"data"/"objaverse"/"scales.json").read_text())
    have = {p.stem for p in MESHES.glob("*.glb")}

    candidates = sorted(u for u in scales if u in paths and u not in have)
    random.Random(0).shuffle(candidates)
    chosen = candidates[:N]
    print(f"{len(have)} meshes already here, downloading {len(chosen)} of {len(candidates)} candidates")

    def one(u):
        dest = MESHES/f"{u}.glb"
        if dest.exists():
            return u, None
        try:
            fetch(paths[u], dest)
            return u, None
        except Exception as e:                                   # a few objects are gone from the hub
            return u, f"{type(e).__name__}: {e}"

    failed = []
    with cf.ThreadPoolExecutor(16) as pool:
        for i, (u, err) in enumerate(pool.map(one, chosen), 1):
            if err:
                failed.append(u)
            if i % 100 == 0:
                print(f"  {i}/{len(chosen)}", flush=True)
    print(f"done, {len(failed)} failed")

    # names, from the metadata shards (one per glbs/000-xxx folder)
    ann_file = ROOT/"data"/"objaverse"/"annotations.json"
    ann = json.loads(ann_file.read_text())
    by_shard = {}
    for u in chosen:
        if u not in failed and u not in ann:
            by_shard.setdefault(paths[u].split("/")[1], []).append(u)
    for shard, uids in by_shard.items():
        rel = f"metadata/{shard}.json.gz"
        if not (CACHE/rel).exists():
            (CACHE/"metadata").mkdir(exist_ok=True)
            fetch(rel, CACHE/rel)
        meta = json.load(gzip.open(CACHE/rel))
        for u in uids:
            m = meta.get(u, {})
            ann[u] = {"name": m.get("name", ""), "tags": [t["name"] for t in m.get("tags", [])]}
    ann_file.write_text(json.dumps(ann, indent=1))
    print(f"{len(ann)} names in {ann_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
