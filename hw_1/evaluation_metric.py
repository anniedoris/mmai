import cadquery as cq
from typing import Tuple, Union
import numpy as np
import os

def generate_step_from_code_file(code_file: str):
    with open(code_file, "r") as f:
        code = f.read()

    step_path = code_file.replace(".py", ".step")
    code += f'\ncq.exporters.export(solid, r"{step_path}")'

    # Run code to generate STEP file
    exec_globals = {"cq": cq, "__builtins__": __builtins__}
    exec(code, exec_globals)
    
    # Check
    if not os.path.exists(step_path) or os.path.getsize(step_path) == 0:
        raise RuntimeError(f"STEP export failed or empty: {step_path}")
    return step_path

# Note: Compute IOU function is taken from my and my lab's prior work: https://github.com/anniedoris/CAD-Coder/blob/main/scripts/compute_iou.py
def compute_iou(source_step_path : str, target_step_path : str) -> Tuple[cq.Workplane, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""

    source = cq.importers.importStep(source_step_path)
    target = cq.importers.importStep(target_step_path)

    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    Rs = np.zeros((4,3,3))
    Rs[0] = I_v_target @ I_v_source.T

    for i in range(3):
        # all possible 2 out of 3 permutations
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

    best_IOU = 0.0
    best_T = None
    for i in range(4):
        T = np.zeros([4,4])
        T[:3,:3] = Rs[i]
        T[-1,-1] = 1
        
        aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
        
        try:
            intersect = aligned_source.intersect(normalized_target)
            union = aligned_source.fuse(normalized_target)
            
            IOU = intersect.Volume() / union.Volume()
        except: #handle cases where IOU is undefined
            IOU = 0.0
        
        if IOU > best_IOU:
            best_IOU = IOU
            best_T = T

    if best_T is not None:
        aligned_source = normalized_source.transformGeometry(cq.Matrix(best_T.tolist())).scale(s_target).translate(c_target)
        return best_IOU
    else:
        aligned_source = None
        return best_IOU
    
def evaluation_metric(predictions, ground_truths):
    ious = []
    for p, gt in zip(predictions, ground_truths):
        p_step_path = generate_step_from_code_file(p)
        gt_step_path = generate_step_from_code_file(gt)
        iou = compute_iou(p_step_path, gt_step_path)
        ious.append(iou)
    avg_iou = sum(ious) / len(ious)
    return avg_iou, ious

def main():
    
    # Set up test examples, first pairs are different CADs (IOU should be less than 1) and second two are the same CADs (IOU equals 1, perfect score)
    predictions = ["metric_test/sample_a.py", "metric_test/sample_a.py"]
    ground_truths = ["metric_test/sample_b.py", "metric_test/sample_a.py"]
    
    avg_iou, ious = evaluation_metric(predictions, ground_truths)
    print(f"Average IOU: {avg_iou}")
    print(f"Individual IOUs: {ious}")
    return

if __name__ == "__main__":
    main()