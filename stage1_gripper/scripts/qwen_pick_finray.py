"""Ask the local Qwen3-VL-2B to choose between our 2-finger and 4-finger fin-ray
gripper, and to say why in one line.

Each object is shown two images -- the Objaverse dataset thumbnail and a two-view
render of the mesh -- plus its dataset name and measured size.

Every object is asked twice, with the two grippers listed in both orders, so a
constant answer can be told apart from option-order bias.

Runs in the `mlxvlm` conda env; the notebook calls it as a subprocess.

    python stage1_gripper/scripts/qwen_pick_finray.py <stage1_gripper/results/demo_dir> <out.json>
"""
import json, pathlib, re, sys

from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/Qwen3-VL-2B-Instruct-4bit"

TWO  = "  2-finger - two fingers close from opposite sides\n"
FOUR = "  4-finger - four fingers close from four sides, caging the object\n"

TEMPLATE = """This object is called "{name}". The first image is its dataset thumbnail; the second shows two views of the 3D mesh.
It measures {short:.0f} mm across its shortest axis and {long:.0f} mm across its longest.

A robot must pick it up with one of two fin-ray grippers. The fingers are identical -- only how many close on the object differs:
{options}
Which should it use? Answer in exactly two lines and nothing else:
gripper: 2-finger or 4-finger
reason: one short sentence"""


def parse(reply):
    """The choice is on the first line; the reason is the sentence after it."""
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    head = re.sub(r"^gripper:\s*", "", lines[0], flags=re.I) if lines else ""
    gripper = ("4-finger" if re.search(r"\b4[\s-]*finger", head, re.I) else
               "2-finger" if re.search(r"\b2[\s-]*finger", head, re.I) else None)
    rest = " ".join(lines[1:]) if len(lines) > 1 else ""
    rest = re.sub(r"^reason:\s*", "", rest, flags=re.I)
    reason = re.split(r"(?<=[.!?])\s", rest)[0] if rest else ""
    return gripper, reason.strip()[:200]


root, out_file = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
objects = json.loads((root/"objects.json").read_text())

model, processor = load(MODEL)
config = load_config(MODEL)


def ask(key, o, options):
    images = [p for p in [root/"thumbs"/f"{key}.jpg", root/"renders"/f"{key}.png"] if p.exists()]
    question = TEMPLATE.format(name=o["name"], short=o["short_mm"],
                               long=o["long_mm"], options=options)
    prompt = apply_chat_template(processor, config, question, num_images=len(images))
    reply = generate(model, processor, prompt, image=[str(p) for p in images],
                     max_tokens=90, temperature=0.0, verbose=False).text.strip()
    return (*parse(reply), reply, len(images))


picks = {}
for key, o in objects.items():
    gripper, reason, raw, n_img = ask(key, o, TWO + FOUR)     # 2-finger listed first
    swapped, _, _, _            = ask(key, o, FOUR + TWO)     # 4-finger listed first
    picks[key] = {"gripper": gripper, "reason": reason, "swapped_order": swapped,
                  "n_images": n_img, "raw": raw[:300]}
    print(f"{key}  {o['name'][:24]:24s} {str(gripper):9s} "
          f"(swapped: {str(swapped):9s}) | {reason}", flush=True)

out_file.write_text(json.dumps(picks, indent=1))
print("saved", out_file)
