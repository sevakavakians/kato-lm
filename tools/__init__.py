"""Tools package for KATO hierarchical learning, hardware analysis, and utilities."""

# Core client and utilities
from .kato_client import KATOClient
from .hardware_analyzer import HardwareAnalyzer
from .streaming_dataset_loader import (
    StreamingDatasetLoader,
    WorkerHealthMonitor,
    recommend_dataset_configuration,
    train_from_streaming_dataset,
    train_from_streaming_dataset_parallel
)

# Hierarchical learning components
from .hierarchical_learning import (
    HierarchicalConceptLearner,
    HierarchicalNode,
    CorpusSegmenter,
    TokenProcessor,
    TokenDecoder,
    TrainingManifest,
    list_available_manifests,
    load_latest_manifest,
    create_training_run_nodes,
    LearningTracker,
    TrainingCheckpoint,
    train_hierarchical_single_pass,
    transfer_all_names,
    transfer_threshold,
    transfer_top_n,
    transfer_weighted,
    transfer_predictions,
)

# Hierarchical builder (v2.0 layer-based API)
from .hierarchical_builder import (
    HierarchicalLayer,
    HierarchicalBuilder,
    HierarchicalModel,
    process_chunk_at_layer,
    accumulate_in_stm,
    learn_from_stm,
    extract_prediction_field
)

# Profiling and Performance Analysis (2.0)
from .profiling_engine import ProfilingEngine, ProfilingReport, ResourceSnapshot
from .storage_estimator import StorageEstimator, StorageEstimate, LevelStorageEstimate
from .hardware_analyzer_v2 import (
    HardwareAnalyzerV2,
    HardwareReportV2,
    ClickHouseBenchmark,
    DiskIOBenchmark,
    NetworkBenchmark,
    GPUInfo
)
from .scaling_analyzer import (
    ScalingAnalyzer,
    ScalingMeasurement,
    ScalingPrediction,
    ScalingAnalysisReport,
    ComplexityCurve
)
from .hardware_recommender import (
    HardwareRecommender,
    HardwareRecommendation,
    CPUSpec,
    MemorySpec,
    StorageSpec,
    CloudInstance
)
from .training_history import (
    TrainingHistory,
    TrainingRun,
    EstimationAccuracy
)
from .training_estimator import (
    TrainingEstimator,
    TrainingTimeEstimate
)
from .training_snapshot import (
    TrainingRunSnapshot,
    NodeSnapshot
)
from .training_comparison import (
    plot_configuration_table,
    plot_frequency_comparison,
    plot_hierarchy_utilization,
    plot_storage_breakdown,
    plot_performance_scatter,
    plot_scaling_analysis,
    plot_efficiency_metrics,
    plot_configuration_heatmap,
    plot_hierarchy_alpha_comparison,
    plot_alpha_vs_patterns_scatter,
    find_optimal_configuration,
    print_comparison_summary
)

__all__ = [
    # Client and utilities
    'KATOClient',
    'HardwareAnalyzer',
    'StreamingDatasetLoader',
    'WorkerHealthMonitor',
    'recommend_dataset_configuration',
    'train_from_streaming_dataset',
    'train_from_streaming_dataset_parallel',
    # Hierarchical learning
    'HierarchicalConceptLearner',
    'HierarchicalNode',
    'CorpusSegmenter',
    'TokenProcessor',
    'TokenDecoder',
    'TrainingManifest',
    'list_available_manifests',
    'load_latest_manifest',
    'create_training_run_nodes',
    'LearningTracker',
    'TrainingCheckpoint',
    'train_hierarchical_single_pass',
    'transfer_all_names',
    'transfer_threshold',
    'transfer_top_n',
    'transfer_weighted',
    'transfer_predictions',
    # Hierarchical builder (v2.0 layer-based API)
    'HierarchicalLayer',
    'HierarchicalBuilder',
    'HierarchicalModel',
    'process_chunk_at_layer',
    'accumulate_in_stm',
    'learn_from_stm',
    'extract_prediction_field',
    # Profiling and Performance Analysis (2.0)
    'ProfilingEngine',
    'ProfilingReport',
    'ResourceSnapshot',
    'StorageEstimator',
    'StorageEstimate',
    'LevelStorageEstimate',
    'HardwareAnalyzerV2',
    'HardwareReportV2',
    'ClickHouseBenchmark',
    'DiskIOBenchmark',
    'NetworkBenchmark',
    'GPUInfo',
    'ScalingAnalyzer',
    'ScalingMeasurement',
    'ScalingPrediction',
    'ScalingAnalysisReport',
    'ComplexityCurve',
    'HardwareRecommender',
    'HardwareRecommendation',
    'CPUSpec',
    'MemorySpec',
    'StorageSpec',
    'CloudInstance',
    'TrainingHistory',
    'TrainingRun',
    'EstimationAccuracy',
    'TrainingEstimator',
    'TrainingTimeEstimate',
    # Training comparison and snapshots
    'TrainingRunSnapshot',
    'NodeSnapshot',
    'plot_configuration_table',
    'plot_frequency_comparison',
    'plot_hierarchy_utilization',
    'plot_storage_breakdown',
    'plot_performance_scatter',
    'plot_scaling_analysis',
    'plot_efficiency_metrics',
    'plot_configuration_heatmap',
    'plot_hierarchy_alpha_comparison',
    'plot_alpha_vs_patterns_scatter',
    'find_optimal_configuration',
    'print_comparison_summary',
]
