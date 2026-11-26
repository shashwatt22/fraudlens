param(
    [Parameter(Mandatory)] [string] $Python,
    [Parameter(Mandatory)] [string] $RawData,
    [Parameter(Mandatory)] [string] $V1Artifacts,
    [ValidateSet('baseline', 'boosting', 'matched', 'patterns', 'finalize', 'evaluate')] [string] $Stage = 'baseline',
    [int] $Threads = 6,
    [string] $OutputRoot = 'artifacts/v2'
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$priorPythonPath = $env:PYTHONPATH
$extraDependencies = Join-Path $project '.venv\Lib\site-packages'
if (Test-Path -LiteralPath $extraDependencies) {
    $env:PYTHONPATH = if ($priorPythonPath) { "$extraDependencies;$priorPythonPath" } else { $extraDependencies }
}

function Invoke-V2 {
    & $Python -B -u @args --output $OutputRoot
    if ($LASTEXITCODE -ne 0) { throw "V2 command failed with exit code $LASTEXITCODE" }
}

Push-Location -LiteralPath $project
try {
    switch ($Stage) {
        'baseline' {
            Invoke-V2 -m src.v2.experiment --raw $RawData --v1 $V1Artifacts --threads $Threads --trials rich v1 history unweighted
        }
        'boosting' {
            Invoke-V2 -m src.v2.challengers --raw $RawData --threads $Threads --candidates xgb_advanced lgbm catboost lgbm_leafwise catboost_regularized xgb_advanced_depth8 xgb_advanced_unweighted
            Invoke-V2 -m src.v2.experiment --raw $RawData --v1 $V1Artifacts --threads $Threads --trials no_vesta no_anonymous basic_native no_identity
        }
        'matched' {
            Invoke-V2 -m src.v2.challengers --raw $RawData --threads $Threads --candidates dummy logistic logistic_l2 gnb gnb_smooth knn knn_neighbors svm svm_l2 decision_tree decision_tree_depth random_forest random_forest_depth adaboost adaboost_depth numpy_linear numpy_linear_l2 numpy_relu numpy_relu_wide torch_mlp torch_mlp_regularized
        }
        'patterns' {
            Invoke-V2 -m src.v2.patterns --threads $Threads
            Invoke-V2 -m src.v2.validation_report
        }
        'finalize' {
            Invoke-V2 -m src.v2.final prepare --threads $Threads
        }
        'evaluate' {
            Invoke-V2 -m src.v2.final evaluate --raw $RawData --v1 $V1Artifacts --threads $Threads
            Invoke-V2 -m src.v2.explanation
        }
    }
} finally {
    Pop-Location
    $env:PYTHONPATH = $priorPythonPath
}
