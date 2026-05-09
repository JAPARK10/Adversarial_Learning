import torch
import os

files = {
    'Baseline 16': 'RFIDDataSet/processed/geometric_data_processed.pt',
    'Exp1 (No SS)': 'RFIDDataSet/processed/geometric_data_exp1_no_ss.pt',
    'Exp2 (Unidir)': 'RFIDDataSet/processed/geometric_data_exp2_unidir.pt',
    'Exp3 (Unidir+No SS)': 'RFIDDataSet/processed/geometric_data_exp3_unidir_no_ss.pt',
}

for label, path in files.items():
    if not os.path.exists(path):
        print(f"{label}: NOT FOUND")
        continue
    data, slices = torch.load(path, weights_only=False)
    participants = sorted(data.participant.unique().tolist())
    n_graphs = len(slices['y']) - 1
    # edges in first graph
    ei_start = slices['edge_index'][0].item()
    ei_end = slices['edge_index'][1].item()
    n_edges_first = ei_end - ei_start
    print(f"{label}:")
    print(f"  Participant IDs: {participants} ({len(participants)} unique)")
    print(f"  Total graphs: {n_graphs}")
    print(f"  Edges in graph[0]: {n_edges_first}")
    print()
