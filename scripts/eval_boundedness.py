import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import load_results


def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    print("=" * 60)
    print("BOUNDEDNESS VERIFICATION")
    print("=" * 60)
    
    tasks = ["boundedness", "recovery", "timescale", "coupled"]
    all_passed = True
    
    for task in tasks:
        results = load_results(results_dir / "stage1" / task)
        if results is None:
            print(f"{task}: NO RESULTS FOUND")
            continue
        
        violations = 0
        for r in results:
            affect_history = r["affect_history"]
            max_norm = np.max(np.abs(affect_history))
            if max_norm > 1.0:
                violations += 1
        
        if violations == 0:
            print(f"{task}: PASS (0/{len(results)} violations)")
        else:
            print(f"{task}: FAIL ({violations}/{len(results)} violations)")
            all_passed = False
    
    print()
    if all_passed:
        print("All boundedness tests PASSED")
    else:
        print("Some boundedness tests FAILED")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
