import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Statevector, SparsePauliOp

# Rebuild your feature‐map & ansatz
nq, reps = 3, 3
fmap   = ZZFeatureMap(feature_dimension=nq, reps=reps)
ansatz = TwoLocal(
    num_qubits=nq,
    rotation_blocks=['ry'],
    entanglement_blocks='cz',
    reps=reps,
)

# Make a single template circuit
qc_template = QuantumCircuit(nq)
qc_template.append(fmap,   range(nq))
qc_template.append(ansatz, range(nq))

# Your measurement operator (Z⊗…⊗Z)
Zop = SparsePauliOp.from_list([('Z'*nq, 1)])

# Load the optimized weights
opt_w = np.load('opt_w.npy')

def predict_probs(X):
    out = []
    for x in X:
        # build one map for fmap params, one for ansatz params
        bind_map = {p: float(v) for p, v in zip(fmap.parameters,   x)}
        bind_map.update({p: float(v) for p, v in zip(ansatz.parameters, opt_w)})
        # assign all at once
        qc = qc_template.assign_parameters(bind_map)
        sv = Statevector.from_instruction(qc)
        ev = sv.expectation_value(Zop).real
        # convert ⟨Z⟩ ∈ [–1,1] to probability ∈ [0,1]
        out.append((1 - ev) / 2)
    return np.array(out)

def main():
    # load fresh data
    df = pd.read_csv('DIA_trainingset_RDKit_descriptors.csv')
    # drop the two non‐numeric columns
    df.drop(columns=['Label', 'SMILES'], inplace=True, errors='ignore')
    # now everything left is numeric
    X = df.values

    p = predict_probs(X)
    print("Quantum model probabilities:", p)

if __name__ == '__main__':
    main()