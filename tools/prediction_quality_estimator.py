#!/usr/bin/env python3
"""
Prediction Quality Estimator - Estimate text generation quality from training snapshots

This module computes metrics to predict how well a trained KATO hierarchy
will perform at text generation, WITHOUT actually running generation tests.

Key metrics:
- Zipf quality (level-specific alpha targets)
- Composition quality (orphan rate, coverage)
- Prediction quality (fan-out, predictive_information)
- Hierarchical consistency (frequency correlation)
- Overall Hierarchical Generation Readiness (HGR) score

Usage:
    from tools import TrainingRunSnapshot, compute_generation_readiness_score

    snapshot = TrainingRunSnapshot.load('snapshots/run_001.json')
    hgr_result = compute_generation_readiness_score(snapshot)

    print(f"Overall HGR Score: {hgr_result['overall_score']}/100")
    print(f"Health Status: {hgr_result['health_status']}")
    for rec in hgr_result['recommendations']:
        print(f"  - {rec}")

Author: KATO Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# LEVEL-SPECIFIC TARGETS (from HIERARCHICAL_GENERATION_ARCHITECTURE.md)
# =============================================================================

# Zipfian alpha targets by level
ZIPF_TARGETS = {
    'node0': {'min': 1.0, 'max': 1.5, 'ideal': 1.25},  # Classic Zipfian
    'node1': {'min': 0.8, 'max': 1.2, 'ideal': 1.0},   # Moderate power-law
    'node2': {'min': 0.5, 'max': 1.0, 'ideal': 0.75},  # Weaker power-law
    'node3': {'min': 0.2, 'max': 0.5, 'ideal': 0.35},  # Nearly uniform
}

# Fan-out targets by level (expected number of predictions)
FANOUT_TARGETS = {
    'node0': {'min': 50, 'max': 150, 'ideal': 100},    # 0.1% of ~100K patterns
    'node1': {'min': 5, 'max': 20, 'ideal': 10},       # 0.1% of ~10K patterns
    'node2': {'min': 2, 'max': 10, 'ideal': 5},        # 0.5% of ~1K patterns
    'node3': {'min': 1, 'max': 5, 'ideal': 2},         # 2% of ~100 patterns
}

# Orphan rate thresholds (lower is better)
ORPHAN_THRESHOLDS = {
    'excellent': 0.10,  # < 10% orphans
    'good': 0.20,       # < 20% orphans
    'warning': 0.30,    # < 30% orphans
    # > 30% = poor
}

# Coverage thresholds (higher is better)
COVERAGE_THRESHOLDS = {
    'excellent': 0.80,  # > 80% used by parent
    'good': 0.70,       # > 70% used
    'warning': 0.50,    # > 50% used
    # < 50% = poor
}


# =============================================================================
# ZIPF QUALITY METRICS
# =============================================================================

def compute_zipf_quality_score(
    snapshot: Any,  # TrainingRunSnapshot
    node_name: str
) -> Dict[str, Any]:
    """
    Compute Zipf quality with level-specific targets.

    Different hierarchical levels have different expected Zipfian alpha values:
    - node0: α ≈ 1.0-1.5 (classic Zipfian for natural language)
    - node1: α ≈ 0.8-1.2 (moderate power-law)
    - node2: α ≈ 0.5-1.0 (weaker power-law)
    - node3: α ≈ 0.2-0.5 (nearly uniform distribution)

    Args:
        snapshot: TrainingRunSnapshot instance
        node_name: Node name (e.g., 'node0')

    Returns:
        Dict with:
            - score: 0.0-1.0 quality score
            - alpha: Measured alpha value
            - target_range: (min, max) expected range
            - ideal: Ideal alpha value
            - status: 'excellent' | 'good' | 'warning' | 'poor'
            - deviation: Distance from ideal
    """
    node_snapshot = snapshot.nodes.get(node_name)
    if not node_snapshot or node_snapshot.zipf_alpha is None:
        return {
            'score': 0.0,
            'alpha': None,
            'target_range': None,
            'ideal': None,
            'status': 'no_data',
            'deviation': None
        }

    alpha = node_snapshot.zipf_alpha

    # Get level-specific targets
    if node_name not in ZIPF_TARGETS:
        return {
            'score': 0.0,
            'alpha': alpha,
            'target_range': None,
            'ideal': None,
            'status': 'unknown_level',
            'deviation': None
        }

    targets = ZIPF_TARGETS[node_name]
    min_alpha = targets['min']
    max_alpha = targets['max']
    ideal = targets['ideal']

    deviation = abs(alpha - ideal)

    # Score based on position in range
    if min_alpha <= alpha <= max_alpha:
        # Within acceptable range
        max_deviation = max(ideal - min_alpha, max_alpha - ideal)
        score = 1.0 - (deviation / max_deviation)
        status = 'excellent' if score > 0.8 else 'good'
    else:
        # Outside acceptable range
        if alpha < min_alpha:
            # Too low: linear penalty
            score = max(0.0, alpha / min_alpha)
            status = 'poor' if score < 0.5 else 'warning'
        else:
            # Too high: inverse penalty
            score = max(0.0, min(1.0, max_alpha / alpha))
            status = 'poor' if score < 0.5 else 'warning'

    return {
        'score': float(score),
        'alpha': float(alpha),
        'target_range': (min_alpha, max_alpha),
        'ideal': ideal,
        'status': status,
        'deviation': float(deviation)
    }


# =============================================================================
# FAN-OUT QUALITY METRICS
# =============================================================================

def compute_fanout_quality_score(
    snapshot: Any,  # TrainingRunSnapshot
    node_name: str
) -> Dict[str, Any]:
    """
    Compute fan-out quality with level-specific targets.

    Expected fan-out (from cascading filter analysis):
    - node0: ~100 predictions (manageable search space)
    - node1: ~10 predictions (constrained by node0)
    - node2: ~5 predictions (constrained by node1)
    - node3: ~2 predictions (highly constrained)

    Too few predictions = repetitive, boring generation
    Too many predictions = chaotic, incoherent generation

    Args:
        snapshot: TrainingRunSnapshot instance
        node_name: Node name (e.g., 'node0')

    Returns:
        Dict with:
            - score: 0.0-1.0 quality score
            - mean_fanout: Average number of predictions
            - target_range: (min, max) expected range
            - ideal: Ideal fan-out value
            - status: 'excellent' | 'good' | 'warning' | 'poor' | 'no_data'
    """
    node_snapshot = snapshot.nodes.get(node_name)
    if not node_snapshot or not node_snapshot.prediction_samples:
        return {
            'score': 0.0,
            'mean_fanout': None,
            'target_range': None,
            'ideal': None,
            'status': 'no_data'
        }

    fanout_stats = node_snapshot.prediction_samples.get('fanout', {})
    mean_fanout = fanout_stats.get('mean', 0)

    # Get level-specific targets
    if node_name not in FANOUT_TARGETS:
        return {
            'score': 0.0,
            'mean_fanout': mean_fanout,
            'target_range': None,
            'ideal': None,
            'status': 'unknown_level'
        }

    targets = FANOUT_TARGETS[node_name]
    min_fanout = targets['min']
    max_fanout = targets['max']
    ideal = targets['ideal']

    # Score based on position in range
    if min_fanout <= mean_fanout <= max_fanout:
        # Within acceptable range
        deviation = abs(mean_fanout - ideal)
        max_deviation = max(ideal - min_fanout, max_fanout - ideal)
        score = 1.0 - (deviation / max_deviation)
        status = 'excellent' if score > 0.8 else 'good'
    else:
        # Outside acceptable range
        if mean_fanout < min_fanout:
            # Too few predictions (repetitive)
            score = mean_fanout / min_fanout if min_fanout > 0 else 0.0
            status = 'poor'
        else:
            # Too many predictions (chaotic)
            score = max_fanout / mean_fanout if mean_fanout > 0 else 0.0
            status = 'warning'
        score = max(0.0, min(1.0, score))

    return {
        'score': float(score),
        'mean_fanout': float(mean_fanout),
        'target_range': (min_fanout, max_fanout),
        'ideal': ideal,
        'status': status
    }


# =============================================================================
# COMPOSITION QUALITY METRICS
# =============================================================================

def compute_composition_quality_score(
    snapshot: Any,  # TrainingRunSnapshot
    node_name: str
) -> Dict[str, Any]:
    """
    Compute composition quality from orphan rate and coverage.

    Orphan rate: % of patterns with no parents (not used by higher levels)
    Coverage: % of patterns used by parent level

    Low orphan rate + high coverage = good hierarchical composition
    High orphan rate + low coverage = poor composition, limited hierarchy benefits

    Args:
        snapshot: TrainingRunSnapshot instance
        node_name: Node name (e.g., 'node0')

    Returns:
        Dict with:
            - score: 0.0-1.0 overall composition quality
            - orphan_rate: % orphans (0-1)
            - orphan_score: Quality based on orphan rate
            - coverage: % used by parent (0-1)
            - coverage_score: Quality based on coverage
            - status: 'excellent' | 'good' | 'warning' | 'poor' | 'no_data'
    """
    node_snapshot = snapshot.nodes.get(node_name)
    if not node_snapshot:
        return {
            'score': 0.0,
            'orphan_rate': None,
            'orphan_score': 0.0,
            'coverage': None,
            'coverage_score': 0.0,
            'status': 'no_data'
        }

    orphan_rate = node_snapshot.orphan_rate if node_snapshot.orphan_rate is not None else 1.0
    coverage = node_snapshot.coverage_to_parent if node_snapshot.coverage_to_parent is not None else 0.0

    # Orphan quality (inverse of orphan rate)
    orphan_score = 1.0 - min(orphan_rate, 1.0)

    # Coverage quality (direct)
    coverage_score = coverage

    # Combined score (weighted)
    composition_score = 0.6 * orphan_score + 0.4 * coverage_score

    # Determine status
    if composition_score > 0.8:
        status = 'excellent'
    elif composition_score > 0.6:
        status = 'good'
    elif composition_score > 0.4:
        status = 'warning'
    else:
        status = 'poor'

    return {
        'score': float(composition_score),
        'orphan_rate': float(orphan_rate),
        'orphan_score': float(orphan_score),
        'coverage': float(coverage),
        'coverage_score': float(coverage_score),
        'status': status
    }


# =============================================================================
# HIERARCHICAL CONSISTENCY METRICS
# =============================================================================

def compute_hierarchical_consistency_score(
    snapshot: Any,  # TrainingRunSnapshot
    node_name: str
) -> Dict[str, Any]:
    """
    Compute hierarchical consistency from frequency correlation.

    Validates that parent frequencies align with sum of child frequencies.
    High correlation = good hierarchy integrity
    Low correlation = structural issues, poor composition

    Args:
        snapshot: TrainingRunSnapshot instance
        node_name: Node name (e.g., 'node1' to check node1 vs node0)

    Returns:
        Dict with:
            - score: 0.0-1.0 consistency score
            - freq_correlation: Spearman correlation
            - freq_compression_ratio: mean(parent) / mean(children)
            - status: 'excellent' | 'good' | 'warning' | 'poor' | 'no_data'
    """
    node_snapshot = snapshot.nodes.get(node_name)
    if not node_snapshot:
        return {
            'score': 0.0,
            'freq_correlation': None,
            'freq_compression_ratio': None,
            'status': 'no_data'
        }

    freq_corr = node_snapshot.freq_correlation_to_children
    freq_ratio = node_snapshot.freq_compression_ratio

    if freq_corr is None:
        return {
            'score': 0.0,
            'freq_correlation': None,
            'freq_compression_ratio': freq_ratio,
            'status': 'no_data'
        }

    # Score based on correlation strength
    # Target: > 0.7 (strong positive correlation)
    if freq_corr >= 0.7:
        score = 1.0
        status = 'excellent'
    elif freq_corr >= 0.5:
        score = 0.5 + (freq_corr - 0.5) * 2.5  # Linear 0.5-1.0
        status = 'good'
    elif freq_corr >= 0.3:
        score = (freq_corr - 0.3) * 2.5  # Linear 0.0-0.5
        status = 'warning'
    else:
        score = 0.0
        status = 'poor'

    return {
        'score': float(score),
        'freq_correlation': float(freq_corr),
        'freq_compression_ratio': float(freq_ratio) if freq_ratio else None,
        'status': status
    }


# =============================================================================
# PREDICTIVE INFORMATION QUALITY
# =============================================================================

def compute_predictive_information_quality(
    snapshot: Any,  # TrainingRunSnapshot
    node_name: str
) -> Dict[str, Any]:
    """
    Compute quality of predictive_information scores from sampled predictions.

    Predictive information measures how reliably patterns predict their futures.
    Higher values = more reliable predictions.

    Args:
        snapshot: TrainingRunSnapshot instance
        node_name: Node name

    Returns:
        Dict with:
            - score: 0.0-1.0 quality score
            - mean_pi: Mean predictive_information
            - mean_potential: Mean potential score
            - status: 'excellent' | 'good' | 'warning' | 'poor' | 'no_data'
    """
    node_snapshot = snapshot.nodes.get(node_name)
    if not node_snapshot or not node_snapshot.prediction_samples:
        return {
            'score': 0.0,
            'mean_pi': None,
            'mean_potential': None,
            'status': 'no_data'
        }

    pred_samples = node_snapshot.prediction_samples

    pi_stats = pred_samples.get('predictive_information', {})
    potential_stats = pred_samples.get('potential', {})

    mean_pi = pi_stats.get('mean', 0.0)
    mean_potential = potential_stats.get('mean', 0.0)

    # Target: mean PI > 0.5 (moderate to high predictive value)
    if mean_pi >= 0.6:
        score = 1.0
        status = 'excellent'
    elif mean_pi >= 0.4:
        score = 0.5 + (mean_pi - 0.4) * 2.5
        status = 'good'
    elif mean_pi >= 0.2:
        score = (mean_pi - 0.2) * 2.5
        status = 'warning'
    else:
        score = 0.0
        status = 'poor'

    return {
        'score': float(score),
        'mean_pi': float(mean_pi),
        'mean_potential': float(mean_potential),
        'status': status
    }


# =============================================================================
# HIERARCHICAL GENERATION READINESS (HGR) COMPOSITE SCORE
# =============================================================================

def compute_generation_readiness_score(
    snapshot: Any,  # TrainingRunSnapshot
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute overall Hierarchical Generation Readiness (HGR) score.

    Aggregates all quality metrics into a single 0-100 score that predicts
    text generation quality without running actual generation tests.

    Categories (default weights):
        1. Zipf Quality (25%): Level-specific alpha targets
        2. Composition Quality (25%): Orphan rate + coverage
        3. Prediction Quality (25%): Fan-out + predictive_information
        4. Hierarchical Consistency (20%): Frequency correlation
        5. Diversity (5%): Pattern vocabulary richness

    Args:
        snapshot: TrainingRunSnapshot instance
        weights: Optional custom category weights

    Returns:
        Dict with:
            - overall_score: 0-100 HGR score
            - health_status: 'EXCELLENT' | 'GOOD' | 'WARNING' | 'POOR' | 'CRITICAL'
            - category_scores: Dict of category scores (0-1)
            - per_level_scores: Dict of scores per node
            - recommendations: List of actionable recommendations
    """
    # Default weights
    if weights is None:
        weights = {
            'zipf_quality': 0.25,
            'composition_quality': 0.25,
            'prediction_quality': 0.25,
            'hierarchical_consistency': 0.20,
            'diversity': 0.05
        }

    recommendations = []
    per_level_scores = {}

    # Compute scores for each level
    node_names = sorted(snapshot.nodes.keys())

    # Category 1: Zipf Quality (average across levels)
    zipf_scores = []
    for node_name in node_names:
        zipf_result = compute_zipf_quality_score(snapshot, node_name)
        zipf_scores.append(zipf_result['score'])

        if zipf_result['status'] in ['poor', 'warning'] and zipf_result['alpha']:
            recommendations.append(
                f"{node_name}: Zipf α={zipf_result['alpha']:.3f} outside ideal "
                f"range {zipf_result['target_range']}. Consider adjusting chunk_size "
                f"or increasing training data."
            )

    zipf_quality_score = np.mean(zipf_scores) if zipf_scores else 0.0

    # Category 2: Composition Quality (average across levels)
    comp_scores = []
    for node_name in node_names:
        comp_result = compute_composition_quality_score(snapshot, node_name)
        comp_scores.append(comp_result['score'])

        if comp_result['orphan_rate'] and comp_result['orphan_rate'] > 0.3:
            recommendations.append(
                f"{node_name}: High orphan rate ({comp_result['orphan_rate']:.1%}). "
                f"Increase training samples or review document segmentation."
            )

        if comp_result['coverage'] and comp_result['coverage'] < 0.5:
            recommendations.append(
                f"{node_name}: Low coverage ({comp_result['coverage']:.1%}). "
                f"Patterns not being effectively composed by parent level."
            )

    composition_quality_score = np.mean(comp_scores) if comp_scores else 0.0

    # Category 3: Prediction Quality (fan-out + predictive_information)
    pred_scores = []
    for node_name in node_names:
        fanout_result = compute_fanout_quality_score(snapshot, node_name)
        pi_result = compute_predictive_information_quality(snapshot, node_name)

        # Average fan-out and PI scores
        node_pred_score = (fanout_result['score'] + pi_result['score']) / 2
        pred_scores.append(node_pred_score)

        if fanout_result['status'] == 'poor':
            mean_fanout = fanout_result.get('mean_fanout', 0)
            recommendations.append(
                f"{node_name}: Fan-out ({mean_fanout:.1f}) outside healthy range "
                f"{fanout_result['target_range']}. Review recall_threshold setting."
            )

    prediction_quality_score = np.mean(pred_scores) if pred_scores else 0.0

    # Category 4: Hierarchical Consistency (skip node0, check node1-node3)
    consistency_scores = []
    for node_name in node_names[1:]:  # Skip node0
        consistency_result = compute_hierarchical_consistency_score(snapshot, node_name)
        consistency_scores.append(consistency_result['score'])

        if consistency_result['status'] in ['poor', 'warning']:
            corr = consistency_result.get('freq_correlation')
            if corr is not None:
                recommendations.append(
                    f"{node_name}: Low frequency correlation ({corr:.2f}) with children. "
                    f"Check hierarchy architecture or training process."
                )

    hierarchical_consistency_score = np.mean(consistency_scores) if consistency_scores else 0.0

    # Category 5: Diversity (pattern vocabulary richness)
    diversity_scores = []
    for node_name in node_names:
        node_snapshot = snapshot.nodes[node_name]
        if node_snapshot.total_patterns > 0 and node_snapshot.total_observations > 0:
            vocab_richness = node_snapshot.total_patterns / node_snapshot.total_observations
            # Target: 0.3-0.7 (Goldilocks zone)
            if 0.3 <= vocab_richness <= 0.7:
                diversity_score = 1.0 - abs(vocab_richness - 0.5) * 2  # Peak at 0.5
            else:
                diversity_score = max(0.0, 1.0 - abs(vocab_richness - 0.5) * 3)
            diversity_scores.append(diversity_score)

    diversity_score = np.mean(diversity_scores) if diversity_scores else 0.0

    # Aggregate category scores
    category_scores = {
        'zipf_quality': zipf_quality_score,
        'composition_quality': composition_quality_score,
        'prediction_quality': prediction_quality_score,
        'hierarchical_consistency': hierarchical_consistency_score,
        'diversity': diversity_score
    }

    # Weighted composite score
    overall_score = sum(
        category_scores.get(cat, 0.0) * weight
        for cat, weight in weights.items()
    ) * 100  # Convert to 0-100 scale

    # Determine health status
    if overall_score >= 80:
        health_status = 'EXCELLENT'
    elif overall_score >= 60:
        health_status = 'GOOD'
    elif overall_score >= 40:
        health_status = 'WARNING'
    elif overall_score >= 20:
        health_status = 'POOR'
    else:
        health_status = 'CRITICAL'

    # Add general recommendations based on overall score
    if overall_score < 60:
        recommendations.insert(0,
            "Overall HGR score below 60. Review all category scores and address "
            "weak areas before proceeding to generation."
        )

    if not recommendations:
        recommendations.append(
            "No critical issues detected. System appears ready for generation testing."
        )

    return {
        'overall_score': float(overall_score),
        'health_status': health_status,
        'category_scores': category_scores,
        'per_level_scores': per_level_scores,
        'recommendations': recommendations
    }


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def compare_snapshots(
    snapshots: List[Any],  # List[TrainingRunSnapshot]
    metric: str = 'overall_score'
) -> Dict[str, Any]:
    """
    Compare multiple training snapshots and rank by quality.

    Args:
        snapshots: List of TrainingRunSnapshot instances
        metric: Metric to rank by ('overall_score' | 'zipf_quality' | etc.)

    Returns:
        Dict with:
            - rankings: List of (run_id, score) tuples sorted by score
            - best_run: run_id of best configuration
            - worst_run: run_id of worst configuration
            - score_range: (min, max) scores
    """
    results = []

    for snapshot in snapshots:
        hgr = compute_generation_readiness_score(snapshot)

        if metric == 'overall_score':
            score = hgr['overall_score']
        elif metric in hgr['category_scores']:
            score = hgr['category_scores'][metric] * 100  # Scale to 0-100
        else:
            score = 0.0

        results.append((snapshot.run_id, score, hgr))

    # Sort by score descending
    rankings = sorted(results, key=lambda x: x[1], reverse=True)

    if rankings:
        best_run = rankings[0][0]
        worst_run = rankings[-1][0]
        score_range = (rankings[-1][1], rankings[0][1])
    else:
        best_run = None
        worst_run = None
        score_range = (0.0, 0.0)

    return {
        'rankings': [(run_id, score) for run_id, score, _ in rankings],
        'detailed_results': {run_id: hgr for run_id, score, hgr in rankings},
        'best_run': best_run,
        'worst_run': worst_run,
        'score_range': score_range
    }
