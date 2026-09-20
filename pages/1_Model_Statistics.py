"""Paper reference and local experiment dashboard, available without a checkpoint."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from road_damage.statistics import PAPER_URL, PAPER_COMPARISON, PAPER_CLASSES, MODULES, read_run

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Model Statistics · Road Damage Inspector', page_icon='📊', layout='wide')
st.caption('ROAD DAMAGE INSPECTOR / RESEARCH & RESULTS')
st.title('Model statistics & characteristics')
st.write('Explore the published YOLOv8-PD results and the training measurements saved by your project.')
st.info('Paper results describe the authors’ experiments. Your local baseline and PD-inspired variants have their own results; the experimental implementation is not an exact reproduction.')

paper, architecture, local = st.tabs(['Paper results', 'Architecture & setup', 'Your trained model'])

with paper:
    st.subheader('YOLOv8-PD at a glance')
    st.caption('Reported on RDD2022 · Table 5, page 10')
    for col, label, value, delta in zip(st.columns(4),
        ['mAP@50', 'mAP@50–95', 'Parameters', 'Computational cost'],
        ['70.6%', '39.5%', '2.3 M', '6.1 GFLOPs'],
        ['+1.4 percentage points', '+1.5 percentage points', '25.8% fewer', '25.6% less']):
        col.metric(label, value, delta, delta_color='off', border=True)
    st.caption('Changes are relative to the paper’s YOLOv8n baseline. Reductions are calculated from the rounded Table 5 values.')
    st.dataframe(pd.DataFrame(PAPER_COMPARISON), hide_index=True, use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        st.subheader('Detection by damage category')
        st.bar_chart(pd.DataFrame(PAPER_CLASSES).set_index('Damage'), horizontal=True,
                     color=['#A8B8C8', '#087E8B'], stack=False)
        st.caption('Table 7, page 12 · Per-class AP@50. These are published measurements, not predictions from your checkpoint.')
    with right:
        st.subheader('Where it struggles')
        st.write('Potholes have the lowest AP@50 at **53.1%**. The authors show missed small potholes and missed detections in scenes with multiple potholes.')
        st.metric('Reported processing speed', '111.9 FPS', border=True)
        st.caption('Discussion, page 12. Reported test hardware: RTX 2060 (6 GB) and Intel i7-10750H. This is not a speed measurement of your computer; the timing protocol is not fully specified.')

    with st.expander('Second dataset: RoadDamage'):
        st.write('Table 6, page 11 reports a second experiment on 3,321 smartphone images with an 80/20 training–validation split.')
        st.dataframe(pd.DataFrame([
            {'Model':'YOLOv8n', 'mAP@50 (%)':60.1, 'mAP@50–95 (%)':27.2, 'F1':.58},
            {'Model':'YOLOv8-PD', 'mAP@50 (%)':64.2, 'mAP@50–95 (%)':29.3, 'F1':.63},
        ]), hide_index=True, use_container_width=True)
        st.caption('The mAP@50 gain is 4.1 percentage points from the table. This is a separate dataset experiment, not evidence of zero-shot transfer.')
    with st.expander('How to read the metrics and source limitations'):
        st.markdown('''- **mAP@50:** mean average precision across damage classes at an intersection-over-union threshold of 0.50.
- **mAP@50–95:** average over overlap thresholds from 0.50 to 0.95; stricter box placement lowers this score.
- **Precision / recall:** how often detections are correct / how much annotated damage is found.
- **F1:** harmonic mean of precision and recall at an operating threshold.
- **Parameters:** learned model values; **GFLOPs:** estimated computation, not elapsed time.
- **FPS:** throughput under a particular hardware and measurement setup, not a portable performance guarantee.''')
        st.write('The paper contains inconsistencies between prose and tables. This page uses Table 5 for the overall RDD2022 comparison and Table 6 for RoadDamage. Table 7’s per-class mAP@50–95 values do not reconcile with its ALL row, so that breakdown is not charted here. The paper also alternates LKSA/LSKA terminology.')
    st.markdown(f'**Source:** Jiayi Zeng & Han Zhong, *YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n model*, Scientific Reports 14, 12052 (2024). [Read the paper]({PAPER_URL}).')

with architecture:
    st.subheader('What the model does')
    st.write('A one-stage object detector receives a road image and returns bounding boxes, class labels and confidence scores for D00, D10, D20 and D40. It does not estimate damage depth, repair cost or pixel-level crack boundaries.')
    st.code('Road image → Backbone features → Multi-scale neck → Detection head → Boxes + classes', language=None)
    st.dataframe(pd.DataFrame(MODULES), hide_index=True, use_container_width=True)
    st.warning('Your baseline does not include these four custom modules. Only the local pd variant combines all four; the other variants support individual experiments.')
    st.subheader('Published training setup')
    st.dataframe(pd.DataFrame([
        {'Characteristic':'Dataset selected for experiment', 'Paper configuration':'10,000 RDD2022 images; 8,000 train / 2,000 validation'},
        {'Characteristic':'Full RDD2022 dataset', 'Paper configuration':'47,420 images from six countries; over 55,000 damage instances'},
        {'Characteristic':'Input', 'Paper configuration':'640 × 640 RGB'},
        {'Characteristic':'Training', 'Paper configuration':'300 epochs; batch size 16'},
        {'Characteristic':'Optimizer', 'Paper configuration':'SGD; initial learning rate 0.01'},
        {'Characteristic':'Framework', 'Paper configuration':'Python / PyTorch / CUDA 11.8'},
        {'Characteristic':'Hardware', 'Paper configuration':'Intel i7-10750H; NVIDIA RTX 2060, 6 GB'},
    ]), hide_index=True, use_container_width=True)
    st.caption('Source: Datasets and evaluation parameters, page 8. The paper’s mosaic timing description is ambiguous; the project disables mosaic for the final 10 epochs.')
    with st.expander('Implementation differences'):
        st.markdown((ROOT / 'docs/architecture.md').read_text(encoding='utf-8'))

with local:
    st.subheader('Your training run')
    runs = sorted((p for p in (ROOT/'runs/train').glob('*') if p.is_dir()
                   and not p.name.startswith('smoke_') and (p/'results.csv').is_file()),
                  key=lambda p: (p/'results.csv').stat().st_mtime, reverse=True)
    if not runs:
        st.info('No training results found yet. This section fills automatically when a run writes results.csv. The paper tabs remain available before training.')
    else:
        selected = st.selectbox('Experiment', [p.name for p in runs])
        try:
            report = read_run(ROOT/'runs/train'/selected)
        except (OSError, ValueError) as exc:
            st.error(f'Could not read this run: {exc}')
            st.stop()
        history = report['history']
        config = report['configuration']
        if not history:
            st.info('This run has no completed epoch rows yet.')
        else:
            last = history[-1]
            st.caption(f"Variant: {config.get('variant', 'not recorded')} · Latest logged epoch: {int(last['epoch'])} · Source: runs/train/{selected}/results.csv")
            st.write('The values below are **validation metrics from the latest logged epoch**, not a fresh evaluation of best.pt or a test-set score.')
            for col, label, key in zip(st.columns(4), ['Precision', 'Recall', 'mAP@50', 'mAP@50–95'],
                ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']):
                value = last.get(key)
                col.metric(label, f'{value*100:.2f}%' if value is not None else 'Not recorded', border=True)
            frame = pd.DataFrame(history).set_index('epoch')
            metric_names = {'metrics/mAP50(B)':'mAP@50', 'metrics/mAP50-95(B)':'mAP@50–95',
                            'metrics/precision(B)':'Precision', 'metrics/recall(B)':'Recall'}
            present = [key for key in metric_names if key in frame.columns]
            if present:
                st.line_chart((frame[present]*100).rename(columns=metric_names), x_label='Epoch', y_label='Validation metric (%)')
            with st.expander('Training and validation losses'):
                losses = [key for key in frame.columns if key.endswith('_loss')]
                if losses:
                    st.line_chart(frame[losses], x_label='Epoch', y_label='Loss')
            st.caption('The paper used a different split, training duration and hardware. A direct ranking against its scores would not be a controlled comparison.')
        st.subheader('Run configuration')
        if config:
            st.dataframe(pd.DataFrame([{'Setting':k, 'Value':str(v)} for k,v in config.items()]), hide_index=True, use_container_width=True)
        else:
            st.info('Experiment metadata is unavailable. The run may still be in progress.')
        if report['checkpoint_exists']:
            st.code(str(Path(report['checkpoint']).relative_to(ROOT)), language=None)
            if st.button('Inspect checkpoint characteristics'):
                try:
                    from road_damage.inference import load_detector
                    detector = load_detector(report['checkpoint'])
                    params = sum(p.numel() for p in detector.model.parameters())
                    st.metric('Parameters in saved checkpoint', f'{params:,}')
                    st.metric('Checkpoint file size', f"{Path(report['checkpoint']).stat().st_size / 1024**2:.2f} MiB")
                    st.write({'variant':getattr(detector.model, 'variant', 'not recorded'), 'classes':detector.names})
                    st.caption('Loaded from your local checkpoint. GFLOPs and FPS are not inferred from the paper.')
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f'Could not inspect checkpoint: {exc}')
        else:
            st.info('No best.pt checkpoint exists for this run yet.')
        manifest = ROOT/'data/rdd/split_manifest.json'
        if manifest.is_file():
            try:
                dataset = json.loads(manifest.read_text(encoding='utf-8'))
                st.subheader('Currently prepared dataset')
                st.caption('Read from data/rdd/split_manifest.json. This is the current dataset, not a guaranteed snapshot of the selected run’s dataset.')
                st.dataframe(pd.DataFrame([{'Split':s,'Images':n, **dataset.get('objects',{}).get(s,{})}
                                           for s,n in dataset['images'].items()]), hide_index=True, use_container_width=True)
                st.caption(f"Images skipped for unsupported labels: {dataset.get('skipped_images',0):,}. Sequence grouping: {dataset.get('sequence_grouping', False)}.")
            except (OSError, ValueError, KeyError) as exc:
                st.warning(f'Dataset summary unavailable: {exc}')
        st.download_button('Download this run’s statistics', json.dumps(report, indent=2),
                           file_name=f'{selected}_statistics.json', mime='application/json')
