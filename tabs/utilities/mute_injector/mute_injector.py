import os
import sys
import json
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_inject_mutes

logs_path = os.path.join(now_dir, "logs")

EXCLUDE = {"zips", "mute", "mute_spin_v1", "mute_spin_v2", "reference"}


def get_models_list():
    if not os.path.isdir(logs_path):
        return []
    return sorted(
        entry
        for entry in os.listdir(logs_path)
        if os.path.isdir(os.path.join(logs_path, entry))
        and not any(ex in entry for ex in EXCLUDE)
    )


def refresh_models():
    return {"choices": get_models_list(), "__type__": "update"}


def get_current_mute_info(model_name: str):
    """Read the existing filelist.txt and return a summary of mute usage."""
    if not model_name:
        return "Select a model to inspect."

    model_path = os.path.join(logs_path, model_name)
    filelist_path = os.path.join(model_path, "filelist.txt")

    if not os.path.exists(filelist_path):
        return (
            "No filelist.txt found for this model.\n"
            "Feature extraction has not been run yet."
        )

    # Read model_info for context
    info_path = os.path.join(model_path, "model_info.json")
    embedder = "unknown"
    vocoder = "unknown"
    speakers = "unknown"
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        embedder = info.get("embedder_model", "unknown")
        vocoder = info.get("vocoder_architecture", "unknown")
        speakers = str(info.get("speakers_id", "unknown"))

    # Count mute lines and regular lines
    mute_lines = 0
    regular_lines = 0
    with open(filelist_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if os.sep + "mute" + os.sep in line or "/mute/" in line:
                mute_lines += 1
            else:
                regular_lines += 1

    total = mute_lines + regular_lines

    # Estimate current include_mutes value (mute entries ÷ speakers)
    try:
        sp = int(speakers)
        if sp > 0 and mute_lines > 0:
            estimated = mute_lines // sp
            mute_ratio = f"{mute_lines} ({estimated} per speaker × {sp} speaker(s))"
        else:
            mute_ratio = f"{mute_lines}"
    except (ValueError, ZeroDivisionError):
        mute_ratio = f"{mute_lines}"

    return (
        f"Current filelist.txt summary\n"
        f"{'─' * 36}\n"
        f"  Regular training entries : {regular_lines}\n"
        f"  Mute entries             : {mute_ratio}\n"
        f"  Total entries            : {total}\n"
        f"  Speakers                 : {speakers}\n"
        f"  Embedder                 : {embedder}\n"
        f"  Vocoder                  : {vocoder}"
    )


def mute_injector_tab():
    with gr.Column():
        gr.Markdown(
            """
            ## Mute Injector 🔇
            Regenerate `filelist.txt` for an already-extracted model with a different mute count — 
            no need to redo feature extraction.

            > **What are mutes?**  
            > Silent audio entries added to the training list to help the model learn silence / 
            > prevent it from hallucinating audio at quiet regions. The default is **2** per speaker.
            > Set to **0** to remove all mutes.
            """
        )

        with gr.Row():
            model_name = gr.Dropdown(
                label="Model",
                info="Pick an already-extracted model from your logs folder.",
                choices=get_models_list(),
                value=None,
                interactive=True,
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1, min_width=80)

        current_info = gr.Textbox(
            label="Current filelist.txt info",
            value="Select a model above to inspect its current mute settings.",
            lines=8,
            interactive=False,
        )

        with gr.Row():
            include_mutes = gr.Slider(
                label="Mutes per speaker",
                info="Number of mute entries to inject per speaker ID. 0 = no mutes.",
                minimum=0,
                maximum=10,
                step=1,
                value=2,
                interactive=True,
                scale=3,
            )

        inject_btn = gr.Button("Inject Mutes & Regenerate filelist.txt", variant="primary")

        output_info = gr.Textbox(
            label="Result",
            value="",
            lines=8,
            interactive=False,
        )

        # Wire up
        model_name.change(
            fn=get_current_mute_info,
            inputs=[model_name],
            outputs=[current_info],
        )

        refresh_btn.click(
            fn=refresh_models,
            inputs=[],
            outputs=[model_name],
        )

        inject_btn.click(
            fn=run_inject_mutes,
            inputs=[model_name, include_mutes],
            outputs=[output_info],
        ).then(
            fn=get_current_mute_info,
            inputs=[model_name],
            outputs=[current_info],
        )
