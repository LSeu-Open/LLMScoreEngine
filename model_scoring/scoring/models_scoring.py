# ------------------------------------------------------------------------------------------------
# License
# ------------------------------------------------------------------------------------------------

# Copyright (c) 2025 LSeu-Open
# 
# This code is licensed under the MIT License.
# See LICENSE file in the root directory

# ------------------------------------------------------------------------------------------------
# Description
# ------------------------------------------------------------------------------------------------

# This module contains the ModelScorer class for scoring large language models based on:
# - Entity benchmarks
# - Dev benchmarks
# - Community score
# - Technical specifications

# This is the core scoring implementation used by the model_scoring package

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import math
from typing import Optional
import sys
from pathlib import Path
from types import ModuleType
from config import scoring_config as default_scoring_config

# Add project root to sys.path for absolute imports.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ------------------------------------------------------------------------------------------------
# ModelScorer class
# ------------------------------------------------------------------------------------------------

class ModelScorer:
    """
    A class for scoring and evaluating large language models based on multiple criteria.
    
    This class implements a comprehensive scoring system that evaluates models on:
    - Entity benchmarks
    - Dev benchmarks
    - Community engagement
    - Technical specifications
    
    The final score is calculated out of 100 points total, with weights defined
    in the provided configuration.
    
    Attributes:
        model_name (str): Name of the model being scored.
        config (ModuleType): The scoring configuration module.
        entity_score (float): Score for entity benchmarks.
        dev_score (float): Score for dev benchmarks.
        community_score (float): Score based on community engagement.
        technical_score (float): Score based on technical specs.
    """

    def __init__(self, model_name: str = "Unnamed Model", scoring_config: Optional[ModuleType] = None):
        """
        Initialize a ModelScorer instance.
        
        Args:
            model_name (str, optional): Name of the model to score. Defaults to "Unnamed Model".
            scoring_config (ModuleType, optional): A loaded configuration module. 
                                                   If None, uses the default config.
        """
        self.model_name = model_name
        self.config = scoring_config if scoring_config is not None else default_scoring_config

    def calculate_entity_benchmarks(self, benchmark_scores: dict) -> float:
        """
        Calculate entity benchmarks score.
        
        Evaluates performance on core entity benchmarks and scales it to the
        weight defined in SCORE_WEIGHTS.
        
        Args:
            benchmark_scores (dict): Dictionary mapping benchmark names to scores (0-1 range)
            
        Returns:
            float: Weighted score for the category.
        """
        if not benchmark_scores:
            return 0.0
            
        weights = self.config.BENCHMARK_WEIGHTS['entity_benchmarks']
        
        score = 0.0
        total_weight_for_scored_benchmarks = 0.0
        
        for bench_key, result in benchmark_scores.items():
            if bench_key in weights and result is not None:
                score += (result * weights[bench_key])
                total_weight_for_scored_benchmarks += weights[bench_key]
        
        if total_weight_for_scored_benchmarks > 0:
            average_performance = score / total_weight_for_scored_benchmarks
            return average_performance * self.config.SCORE_WEIGHTS['entity_benchmarks']
        return 0.0

    def calculate_dev_benchmarks(self, benchmark_scores: dict) -> float:
        """
        Calculate dev benchmarks score.
        
        Evaluates performance across a wide range of development benchmarks and
        scales it to the weight defined in SCORE_WEIGHTS.
        
        Args:
            benchmark_scores (dict): Dictionary mapping benchmark names to scores (0-1 range).
            
        Returns:
            float: Weighted score for the category.
        """
        if not benchmark_scores:
            return 0.0
            
        weights = self.config.BENCHMARK_WEIGHTS['dev_benchmarks']
        
        current_score = 0.0
        total_weight_of_scored_benchmarks = 0.0
        
        for bench_key, result in benchmark_scores.items():
            if bench_key in weights and result is not None:
                current_score += (result * weights[bench_key])
                total_weight_of_scored_benchmarks += weights[bench_key]

        if total_weight_of_scored_benchmarks > 0:
            average_performance = current_score / total_weight_of_scored_benchmarks
            return average_performance * self.config.SCORE_WEIGHTS['dev_benchmarks']
        return 0.0

    def calculate_community_score(self, lm_sys_arena_elo_rating: Optional[float], hf_score: Optional[float]) -> float:
        """
        Calculate the total community score, scaled to the weight defined in SCORE_WEIGHTS.

        - If both scores are present, they each contribute 50% of the category weight.
        - If only one score is present, it contributes 100% of the category weight.
        
        Args:
            lm_sys_arena_elo_rating (Optional[float]): Model's LMsys Arena ELO rating.
            hf_score (Optional[float]): Model's Hugging Face community score (0-10).
            
        Returns:
            float: Total community score for the category.
        """
        if lm_sys_arena_elo_rating is None and hf_score is None:
            return 0.0
            
        total_score = 0.0
        elo_bounds = self.config.COMMUNITY_SCORE_BOUNDS['lm_sys_arena_score']
        hf_bounds = self.config.COMMUNITY_SCORE_BOUNDS['hf_score']
        category_weight = self.config.SCORE_WEIGHTS['community_score']

        if lm_sys_arena_elo_rating is not None:
            min_elo, max_elo = elo_bounds['min'], elo_bounds['max']
            elo_scale = category_weight if hf_score is None else category_weight / 2.0
            
            if max_elo == min_elo:
                normalized_elo = 0.0 if lm_sys_arena_elo_rating <= min_elo else elo_scale
            else:
                normalized_elo = ((lm_sys_arena_elo_rating - min_elo) / (max_elo - min_elo)) * elo_scale
            
            total_score += max(0.0, min(elo_scale, normalized_elo))
        
        if hf_score is not None:
            min_hf, max_hf = hf_bounds['min'], hf_bounds['max']
            hf_scale = category_weight if lm_sys_arena_elo_rating is None else category_weight / 2.0
            
            # Normalize 0-10 hf_score to the required scale
            normalized_hf = (hf_score / (max_hf - min_hf)) * hf_scale
            total_score += max(0.0, min(hf_scale, normalized_hf))
            
        return round(total_score, 2)

    def _calculate_price_component_score(self, price: Optional[float], price_type: str) -> float:
        """Calculate score for a single price component (input or output)."""
        if price is None:
            return 0.0
        
        params = self.config.TECHNICAL_SCORE_PARAMS[price_type]
        if price <= 0.0:
            # Free models get max points for this component
            return params['max_points']
        if price >= params['high_price_cutoff']:
            return params['high_price_points']

        # Linear interpolation for score
        raw_score = params['intercept'] - (params['coefficient'] * price)
        
        # Ensure score is within defined bounds
        return max(params['high_price_points'], min(params['max_points'], raw_score))

    def _calculate_context_score(self, context_size: Optional[int]) -> float:
        """Calculate score based on context window size, using parameters from config."""
        if context_size is None:
            return 0.0
            
        params = self.config.TECHNICAL_SCORE_PARAMS['context_window']
        if context_size < params['low_cw_cutoff']:
            return params['low_cw_points']
        
        raw_score = params['coefficient'] * math.log(context_size, params['log_base']) + params['intercept']
        return max(params['low_cw_points'], min(params['max_points'], raw_score))

    def calculate_size_perf_ratio(self, benchmark_score: float, param_count: int, architecture: str) -> float:
        """Calculate Model Size vs Performance Ratio score, using parameters from config."""
        if benchmark_score is None or param_count is None or architecture is None:
            return 0.0

        size_params = self.config.TECHNICAL_SCORE_PARAMS['size_perf_ratio']
        
        # 1. Determine Base Size Factor
        base_size_factor = size_params['default_size_factor']
        for tier_limit, factor in sorted(size_params['size_tiers'].items()):
            if param_count < tier_limit:
                base_size_factor = factor
                break

        # 2. Determine Architecture Factor
        arch_factor = self.config.MODEL_ARCHITECTURE_FACTORS.get(architecture.lower(), self.config.MODEL_ARCHITECTURE_FACTORS['default'])

        # 3. Calculate Total Efficiency Factor
        total_efficiency_factor = base_size_factor * arch_factor

        # 4. Calculate Combined Score
        combined_score = (benchmark_score / 100.0) * total_efficiency_factor

        # 5. Calculate Final Points
        points = size_params['base_points'] + (size_params['scaling_factor'] * combined_score)
        return max(size_params['base_points'], min(size_params['max_points'], points))

    def calculate_technical_score(self, input_price: Optional[float], output_price: Optional[float], context_window: Optional[int], benchmark_score: Optional[float], param_count: Optional[int], architecture: Optional[str]) -> float:
        """Calculate technical specifications score."""
        input_price_score = self._calculate_price_component_score(input_price, 'input_price')
        output_price_score = self._calculate_price_component_score(output_price, 'output_price')
        price_score = input_price_score + output_price_score

        context_score = self._calculate_context_score(context_window)
        
        if benchmark_score is not None and param_count is not None and architecture is not None:
            ratio_points = self.calculate_size_perf_ratio(benchmark_score, param_count, architecture)
        else:
            ratio_points = 0.0
        
        total_technical_score = price_score + context_score + ratio_points
        return round(total_technical_score, 2)

    def _display_score_breakdown(self, quiet: bool = False):
        """Display the breakdown of scores in a formatted way."""
        if quiet:
            return
            
        print(f"\n=== Score Breakdown for {self.model_name} ===")
        print(f"{'Entity Benchmarks:':<20} {self.entity_score:.2f} / {self.config.SCORE_WEIGHTS['entity_benchmarks']}")
        print(f"{'Dev Benchmarks:':<20} {self.dev_score:.2f} / {self.config.SCORE_WEIGHTS['dev_benchmarks']}")
        print(f"{'Community Score:':<20} {self.community_score:.2f} / {self.config.SCORE_WEIGHTS['community_score']}")
        print(f"{'Technical Score:':<20} {self.technical_score:.2f} / {self.config.SCORE_WEIGHTS['technical_score']}")
        print("----------------------------------------")
        final_score = self.entity_score + self.dev_score + self.community_score + self.technical_score
        print(f"{'Final Score:':<20} {final_score:.2f} / 100")
        print("========================================")

    def calculate_final_score(self, entity_benchmarks, dev_benchmarks, community_inputs, tech_inputs, quiet: bool = False) -> float:
        """
        Calculate the final score by aggregating all sub-scores.
        
        This is the main scoring function that computes scores for all categories
        and combines them into a final weighted score.
        
        Args:
            entity_benchmarks (dict): Scores for entity-defined benchmarks.
            dev_benchmarks (dict): Scores for dev-defined benchmarks.
            community_inputs (dict): Inputs for community scoring (e.g., ELO, HF score).
            tech_inputs (dict): Inputs for technical scoring (e.g., price, context size).
            quiet (bool): If True, suppresses the score breakdown display.
            
        Returns:
            float: The final aggregated score, rounded to 4 decimal places.
        """
        # Calculate scores for benchmark categories first
        self.entity_score = self.calculate_entity_benchmarks(entity_benchmarks)
        self.dev_score = self.calculate_dev_benchmarks(dev_benchmarks)

        # Calculate a single, unweighted benchmark performance score (0-100)
        # to be used in the technical score calculation.
        all_benchmark_weights = {
            **self.config.BENCHMARK_WEIGHTS['entity_benchmarks'],
            **self.config.BENCHMARK_WEIGHTS['dev_benchmarks']
        }
        all_benchmark_scores = {**entity_benchmarks, **dev_benchmarks}

        score = 0.0
        total_weight = 0.0
        for bench_key, result in all_benchmark_scores.items():
            if bench_key in all_benchmark_weights and result is not None:
                score += (result * all_benchmark_weights[bench_key])
                total_weight += all_benchmark_weights[bench_key]

        overall_benchmark_score = (score / total_weight * 100) if total_weight > 0 else 0.0
        
        # Calculate scores for the remaining categories
        self.community_score = self.calculate_community_score(**community_inputs)
        
        # Pass the overall benchmark score to the technical score calculation
        tech_inputs_with_benchmark = tech_inputs.copy()
        tech_inputs_with_benchmark['benchmark_score'] = overall_benchmark_score
        self.technical_score = self.calculate_technical_score(**tech_inputs_with_benchmark)
        
        # Display the score breakdown
        self._display_score_breakdown(quiet=quiet)
        
        # Aggregate the final score
        final_score = self.entity_score + self.dev_score + self.community_score + self.technical_score
        return round(final_score, 4) 
