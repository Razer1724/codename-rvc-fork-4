import json
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_full_experiment_to_json(log_dir, output_file, max_step=None):
    abs_log_dir = os.path.abspath(log_dir)
    print(f"Targeting directory: {abs_log_dir}")
    if not os.path.exists(abs_log_dir):
        print(f"ERROR: Cannot find folder: {abs_log_dir}")
        return

    target_tags = [
        "loss_avg_100/loss_adv_100",
        "loss_avg_100/loss_disc_100",
        "loss_avg_100/loss_fm_100",
        "loss_avg_100/loss_gen_total_100",
        "loss_avg_100/loss_kl_100",
        "loss_avg_100/loss_spectral_100",
        "grad_avg_100/grad_norm_d_100",
        "grad_avg_100/grad_norm_g_100"
    ]
    ea = EventAccumulator(abs_log_dir, size_guidance={'scalars': 0})

    try:
        ea.Reload()
    except Exception as e:
        print(f"Failed to reload EventAccumulator: {e}")
        return

    data_frames = []
    available_tags = ea.Tags()['scalars']

    for tag in target_tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', tag])

            if max_step is not None:
                df = df[df['step'] <= max_step]

            df = df.drop_duplicates(subset=['step'], keep='last')
            data_frames.append(df.set_index('step'))
        else:
            print(f"Tag not found: {tag}")

    if not data_frames:
        print("No data found. Check if the tfevents files are inside the folder.")
        return

    combined_df = pd.concat(data_frames, axis=1).reset_index()
    combined_df = combined_df.sort_values('step')

    records = combined_df.to_dict(orient='records')
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"Successfully exported {len(records)} rows to {output_file}")

LOG_DIR = r'PATH'
OUTPUT_NAME = 'NAME_OF_OUTPUT.json'
MAX_STEP = 1000

export_full_experiment_to_json(LOG_DIR, OUTPUT_NAME, max_step=MAX_STEP)