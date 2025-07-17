import os
import ast

ML_IMPORTS = [
    'torch', 'transformers', 'sklearn', 'torch_geometric', 'sentence_transformers',
    'stable_baselines3', 'optuna', 'wandb', 'mlflow'
]
REQUIRED_UTILS = [
    'DistributedLogger', 'AdvancedDataValidator'
]
EXPLAINABILITY = ['shap', 'lime', 'attention']
API_DOCS = ['flask_restx', 'fastapi', 'swagger', 'openapi']


def scan_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        tree = ast.parse(code)
        imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
        from_imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        all_imports = set(imports + from_imports)
    except Exception:
        all_imports = set()
    findings = []
    # Check for ML
    if not any(ml in all_imports for ml in ML_IMPORTS):
        findings.append('NO_REAL_ML_IMPORT')
    # Check for logging/validation
    for util in REQUIRED_UTILS:
        if util not in code:
            findings.append(f'MISSING_{util.upper()}')
    # Check for explainability
    if not any(ex in code for ex in EXPLAINABILITY):
        findings.append('MISSING_EXPLAINABILITY')
    # Check for API docs
    if not any(api in code for api in API_DOCS):
        findings.append('MISSING_API_DOCS')
    return findings

def walk_codebase(root='backend'):
    report = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.py'):
                fpath = os.path.join(dirpath, fname)
                findings = scan_file(fpath)
                if findings:
                    report[fpath] = findings
    return report

if __name__ == '__main__':
    report = walk_codebase('backend')
    print('==== ML Perfection Audit Report ====')
    for f, issues in report.items():
        print(f'{f}: {issues}')
    print('==== End of Report ====') 