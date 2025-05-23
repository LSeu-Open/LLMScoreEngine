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
# ModelScorer class
# ------------------------------------------------------------------------------------------------

import math
from typing import Optional

from ..core.constants import LM_SYS_ARENA_SCORE_BOUNDS

class ModelScorer:
    """
    A class for scoring and evaluating large language models based on multiple criteria.
    
    This class implements a comprehensive scoring system that evaluates models on:
    - Entity benchmarks (25 points max)
    - Dev benchmarks (35 points max) 
    - Community engagement (20 points max)
    - Technical specifications (20 points max)
    
    The final score is calculated out of 100 points total.
    
    Attributes:
        model_name (str): Name of the model being scored. Defaults to "Unnamed Model".
        external_score (float): Combined score from entity and dev benchmarks (set externally)
        community_score (float): Score based on community engagement (set externally)
        technical_score (float): Score based on technical specs (set externally)
    """

    def __init__(self, model_name="Unnamed Model"):
        """
        Initialize a ModelScorer instance.
        
        Args:
            model_name (str, optional): Name of the model to score. Defaults to "Unnamed Model".
        """
        self.model_name = model_name
        
    def calculate_entity_benchmarks(self, benchmark_scores: dict) -> float:
        """
        Calculate entity benchmarks score out of 30 points maximum.
        
        Evaluates performance on core entity benchmarks like artificial analysis,
        live code bench, big code models and open LLM evaluations.
        
        Args:
            benchmark_scores (dict): Dictionary mapping benchmark names to scores (0-1 range)
            
        Returns:
            float: Weighted score out of 30 points
        """
        if not benchmark_scores:
            return 0.0
            
        # Define relative weights for each benchmark
        weights = {
            'artificial_analysis': 10, 
            'OpenCompass': 10,         
            'LLM Explorer': 10,        
            'Livebench': 10,           
            'open_llm': 10,            
            'UGI Leaderboard': 10,     
            'big_code_bench': 10,      
            'EvalPlus Leaderboard': 10,
            'Dubesord_LLM': 10,        
            'Open VLM': 10,            
        }
        
        score = 0.0
        total_weight_for_scored_benchmarks = 0.0
        
        # Calculate weighted average of available scores
        for bench_key, result in benchmark_scores.items():
            if bench_key in weights and result is not None:
                score += (result * weights[bench_key])
                total_weight_for_scored_benchmarks += weights[bench_key]
            elif bench_key not in weights and result is not None:
                pass 
                
        # Scale to 30 points maximum if we have scores
        if total_weight_for_scored_benchmarks > 0:
            average_performance = score / total_weight_for_scored_benchmarks
        return 0.0

    def calculate_dev_benchmarks(self, benchmark_scores: dict) -> float:
        """
        Calculate dev benchmarks score out of 30 points maximum.
        
        Evaluates performance across a wide range of development benchmarks as defined in Exploration.md.
        Weights are based on Exploration.md.
        
        Args:
            benchmark_scores (dict): Dictionary mapping benchmark names to scores (0-1 range).
                                     Keys should match those defined in the weights dictionary.
            
        Returns:
            float: Weighted score out of 30 points.
        """
        if not benchmark_scores:
            return 0.0
            
        weights = {
            # General knowledge and reasoning (Total Weight: 28)
            'MMLU': 3,
            'MMLU Pro': 5,
            'BigBenchHard': 3,
            'GPQA diamond': 7,
            'DROP': 3,
            "Humanity's Last Exam": 4,
            'HellaSwag': 3,
            'ARC-C': 3,
            # Instruction following (Total Weight: 12)
            'Wild bench': 3,
            'MT bench': 3,
            'IFEval': 3,
            'Arena Hard': 3,
            # Math (Total Weight: 10)
            'Math': 3,
            'GSM8K': 3,
            'AIME': 4,
            # Coding (Total Weight: 13)
            'HumanEval': 1,
            'MBPP': 1,
            'LiveCodeBench': 4,
            'Aider Polyglot': 2,
            'SWE-Bench': 2,
            'SciCode': 3,
            # Multilingual (Total Weight: 8)
            'MGSM': 2,
            'MMMLU': 2,
            'C-Eval or CMMLU': 2,
            'AraMMLu': 2,
            # Context (Total Weight: 8)
            'LongBench': 2,
            'RULER 128K': 2,
            'RULER 32K': 2,
            'MTOB': 2,
            # Function calling (tool use and agent) (Total Weight: 10)
            'BFCL': 3,
            'AgentBench': 2,
            'Gorilla': 1,
            'ToolBench': 2,
            'MINT': 2,
            # Vision (Total Weight: 8)
            'MMMU': 2,
            'Mathvista': 3,
            'ChartQA': 1,
            'DocVQA': 1,
            'AI2D': 1,
        }
        
        current_score = 0.0
        total_weight_of_scored_benchmarks = 0.0
        
        # Calculate weighted sum of available scores
        for bench_key, result in benchmark_scores.items():
            if bench_key in weights and result is not None:
                current_score += (result * weights[bench_key])
                total_weight_of_scored_benchmarks += weights[bench_key]
            elif bench_key not in weights and result is not None:
                # logger.warning(f"Benchmark key '{bench_key}' not found in defined weights for dev benchmarks.")
                pass 

        if total_weight_of_scored_benchmarks > 0:
            average_performance = current_score / total_weight_of_scored_benchmarks
            return average_performance * 30.0
        return 0.0

    def calculate_external_benchmarks(self, entity_benchmarks, dev_benchmarks=None):
        """
        Calculate total external benchmarks score out of 60 points maximum.
        
        Combines entity benchmarks (30 points) and dev benchmarks (30 points).
        
        Args:
            entity_benchmarks (dict): Entity benchmark scores
            dev_benchmarks (dict, optional): Dev benchmark scores. If None, uses entity_benchmarks
            
        Returns:
            float: Combined external benchmark score out of 60 points
        """
        if dev_benchmarks is None:
            dev_benchmarks = entity_benchmarks  # For backward compatibility
        
        entity_score = self.calculate_entity_benchmarks(entity_benchmarks)
        dev_score = self.calculate_dev_benchmarks(dev_benchmarks)
        return entity_score + dev_score
 
    def calculate_community_score(self, lm_sys_arena_elo_rating: Optional[float], hf_score: Optional[float]) -> Optional[float]:
        """
        Calculate the total community score out of 20 points maximum.

        The calculation depends on which scores are provided:
        - If only `lm_sys_arena_elo_rating` is provided: It's normalized to a 0-20 scale.
        - If only `hf_score` is provided: It's scaled to contribute 0-20 points (assuming input `hf_score` is 0-10).
        - If both are provided: `lm_sys_arena_elo_rating` is normalized to a 0-10 scale,
          and `hf_score` (expected 0-10 points) is added to it.
        The maximum possible score is 20.
        
        Args:
            lm_sys_arena_elo_rating (Optional[float]): Model's LMsys Arena ELO rating.
            hf_score (Optional[float]): Model's Hugging Face community score (expected 0-10 points).
            
        Returns:
            Optional[float]: Total community score. 
                             Returns None if both input scores are None.
        """
        if lm_sys_arena_elo_rating is None and hf_score is None:
            return None
            
        total_score = 0.0
        
        if lm_sys_arena_elo_rating is not None:
            min_elo = LM_SYS_ARENA_SCORE_BOUNDS["MIN"]
            max_elo = LM_SYS_ARENA_SCORE_BOUNDS["MAX"]
            
            # Determine the scale for ELO normalization based on hf_score's presence
            elo_normalization_scale = 20.0 if hf_score is None else 10.0
            
            if max_elo == min_elo: # Avoid division by zero if bounds are misconfigured
                normalized_elo_score = 0.0 if lm_sys_arena_elo_rating <= min_elo else elo_normalization_scale
            else:
                normalized_elo_score = ((lm_sys_arena_elo_rating - min_elo) / (max_elo - min_elo)) * elo_normalization_scale
            
            # Clamp the score between 0 and the determined normalization scale
            normalized_elo_score = max(0.0, min(elo_normalization_scale, normalized_elo_score))
            total_score += normalized_elo_score
        
        if hf_score is not None:
            if lm_sys_arena_elo_rating is None: # HF only case
                hf_contribution = max(0.0, min(20.0, hf_score * 2.0))
            else: # HF is present AND ELO is also present
                hf_contribution = hf_score 
            total_score += hf_contribution
            
        return round(total_score, 2)

    def _calculate_price_score(self, price: float | None) -> float:
        """
        Calculate score based on model's price point (8 points max).
        Uses a linear scale as defined in Exploration.md.

        Args:
            price (float | None): Price per million tokens in USD.

        Returns:
            float: Score from 0.0-8.0 based on price.
                   Returns 0.0 if price is None.
        """
        if price is None:
            return 0.0

        # Apply conditions from Exploration.md for max and min points
        if price <= 0.0:
            return 8.0
        if price >= 20.0:
            return 1.0

        # Formula from Exploration.md: Points = 8.0 - (0.35 * Price)
        # The bounds above (<=0 and >=20) effectively clamp the formula's natural output at its extremes.
        # For prices between 0 and 20, the formula itself will produce values within the 1.0 to 8.0 range.
        # e.g. price=0.01 -> 8.0 - 0.0035 = 7.9965
        #      price=19.99 -> 8.0 - 6.9965 = 1.0035
        
        raw_calculated_score = 8.0 - (0.35 * price)

        # Ensure the score is strictly within 1.0 and 8.0 after calculation, 
        price_score = max(1.0, min(8.0, raw_calculated_score))

        return price_score 

    def _calculate_context_score(self, context_size: int | None) -> float:
        """
        Calculate score based on context window size (6 points max).
        Uses a logarithmic scale (base 2) as defined in Exploration.md.

        Args:
            context_size (int | None): Maximum context window size in tokens.

        Returns:
            float: Score from 0.0-6.0 based on context size.
                   Returns 0.0 if context_size is None.
        """

        if context_size is None:
            return 0.0
        
        if context_size < 8192:
            return 1.0
        else:
            
            raw_score_from_formula = 0.571 * math.log2(context_size) - 5.929
            
            context_score = max(1.0, min(6.0, raw_score_from_formula))
            
            return context_score

    def calculate_size_perf_ratio(self, benchmark_score: float, param_count: int, architecture: str) -> float:
        """
        Calculate Model Size vs Performance Ratio score (6 points max) as defined in Exploration.md.

        This component assesses a model's performance relative to its size and architectural efficiency.
        Points are awarded based on a `Combined Score` using a linear scale.

        Args:
            benchmark_score (float): Average benchmark performance (0-100 scale).
            param_count (int): Actual number of parameters (e.g., 7000000000 for 7B).
            architecture (str): Architecture of the model (e.g., "moe", "ssm", "dense").

        Returns:
            float: Efficiency score from 1.0 to 6.0 points.
        """
        if benchmark_score is None or param_count is None or architecture is None:
            return 0.0 # Or handle as an error/default appropriately

        # 1. Determine Base Size Factor
        if param_count < 3_000_000_000:
            base_size_factor = 1.00
        elif param_count < 10_000_000_000:
            base_size_factor = 0.95
        elif param_count < 30_000_000_000:
            base_size_factor = 0.90
        elif param_count < 80_000_000_000:
            base_size_factor = 0.80
        elif param_count < 200_000_000_000:
            base_size_factor = 0.70
        else: # > 200B
            base_size_factor = 0.60

        # 2. Determine Architecture Factor
        arch_lower = architecture.lower()
        if "moe" in arch_lower:
            architecture_factor = 1.2
        elif "ssm" in arch_lower: # Catching common SSM variants
            architecture_factor = 1.1
        elif arch_lower == "dense" or arch_lower == "dense_transformer":
            architecture_factor = 1.0
        elif "specialized" in arch_lower or "efficient" in arch_lower: # For "Other specialized efficient architectures"
            architecture_factor = 1.1 
        else: # Default to Dense Transformer if not recognized
            architecture_factor = 1.0

        # 3. Calculate Total Efficiency Factor
        total_efficiency_factor = base_size_factor * architecture_factor

        # 4. Calculate Combined Score
        combined_score = (benchmark_score / 100.0) * total_efficiency_factor

        # 5. Calculate Final Points using the linear formula with bounds
        points = max(1.0, min(6.0, 1.0 + (5.0 * combined_score)))

        return points

    def calculate_technical_score(self, price: float | None, context_window: int | None, benchmark_score: float | None, param_count: int | None, architecture: str | None) -> float:
        """
        Calculate technical specifications score out of 20 points maximum.
        
        Combines scores for:
        - Price efficiency (8 points)
        - Context window size (6 points)
        - Model Size vs Performance Ratio (6 points)
        
        Args:
            price (float | None): Price per million tokens in USD.
            context_window (int | None): Maximum context window size in tokens.
            benchmark_score (float | None): Average benchmark performance (0-100 scale) for ratio calculation.
            param_count (int | None): Actual number of parameters for ratio calculation.
            architecture (str | None): Architecture of the model for ratio calculation.
            
        Returns:
            float: Combined technical score out of 20 points.
        """
        price_score = self._calculate_price_score(price)
        context_score = self._calculate_context_score(context_window)
        
        # Directly call the updated calculate_size_perf_ratio method
        # It requires benchmark_score, param_count, and architecture
        if benchmark_score is not None and param_count is not None and architecture is not None:
            ratio_points = self.calculate_size_perf_ratio(benchmark_score, param_count, architecture)
        else:
            ratio_points = 0.0 # Default if necessary info for ratio score is missing
        
        return price_score + context_score + ratio_points

    def calculate_final_score(self):
        """
        Calculate final comprehensive score out of 100 points.
        
        Combines:
        - External benchmarks (60 points)
        - Community score (20 points)
        - Technical score (20 points)
        
        All component scores must be set before calling this method.
        
        Returns:
            float: Final rounded score out of 100 points
            
        Raises:
            ValueError: If any component scores are not set
        """
        # Verify all required scores are set
        if not hasattr(self, 'external_score') or \
           not hasattr(self, 'community_score') or \
           not hasattr(self, 'technical_score'):
            raise ValueError("All component scores must be set before calculating final score")
        
        # Calculate total score
        final_score = self.external_score + self.community_score + self.technical_score
        
        # Print detailed breakdown
        print(f"\n=== Score Breakdown for {self.model_name} ===")
        print(f"External Score:   {self.external_score:>6.2f}/60")
        print(f"Community Score:  {self.community_score:>6.2f}/20")
        print(f"Technical Score:  {self.technical_score:>6.2f}/20")
        print(f"Final Score:      {final_score:>6.2f}/100")
        print("=" * 40)
        
        return round(final_score, 2) 
