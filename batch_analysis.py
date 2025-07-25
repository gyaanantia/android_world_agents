#!/usr/bin/env python3
"""
Batch Episode Replay and Analysis Tool.

This script allows you to replay and analyze multiple episodes at once,
providing comparative analysis and batch processing capabilities.
"""

import argparse
import json
import os
import sys
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Add project root and src to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils import suppress_grpc_logging

# Suppress gRPC verbose logging
suppress_grpc_logging()


class BatchEpisodeAnalyzer:
    """Analyze multiple episodes for patterns and insights."""
    
    def __init__(self, results_dir: str, pattern: str = "*.json"):
        """Initialize the batch analyzer.
        
        Args:
            results_dir: Directory containing result JSON files
            pattern: File pattern to match (default: *.json)
        """
        self.results_dir = results_dir
        self.pattern = pattern
        self.episodes = []
        
    def load_episodes(self) -> List[Dict[str, Any]]:
        """Load all episodes from the results directory.
        
        Returns:
            List of episode data dictionaries
        """
        json_files = glob.glob(os.path.join(self.results_dir, self.pattern))
        print(f"📁 Found {len(json_files)} JSON files in {self.results_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                    episode_data['_file_path'] = json_file
                    episode_data['_file_name'] = os.path.basename(json_file)
                    self.episodes.append(episode_data)
            except Exception as e:
                print(f"⚠️  Failed to load {json_file}: {e}")
                
        print(f"✅ Loaded {len(self.episodes)} episodes successfully\n")
        return self.episodes
        
    def analyze_success_rates(self) -> Dict[str, Any]:
        """Analyze success rates across different dimensions."""
        print("📊 Success Rate Analysis")
        
        # Overall success rate
        total_episodes = len(self.episodes)
        successful_episodes = sum(1 for ep in self.episodes if ep.get('success', False))
        overall_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
        
        print(f"   Overall: {successful_episodes}/{total_episodes} ({overall_success_rate:.1%})")
        
        # Success rate by task
        task_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        for episode in self.episodes:
            task_name = episode.get('task_name', 'Unknown')
            task_stats[task_name]['total'] += 1
            if episode.get('success', False):
                task_stats[task_name]['successful'] += 1
                
        print("\n   By Task:")
        for task, stats in sorted(task_stats.items()):
            rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            print(f"     {task}: {stats['successful']}/{stats['total']} ({rate:.1%})")
            
        # Success rate by model
        model_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        for episode in self.episodes:
            model_name = episode.get('model_name', 'Unknown')
            model_stats[model_name]['total'] += 1
            if episode.get('success', False):
                model_stats[model_name]['successful'] += 1
                
        print("\n   By Model:")
        for model, stats in sorted(model_stats.items()):
            rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            print(f"     {model}: {stats['successful']}/{stats['total']} ({rate:.1%})")
            
        # Success rate by prompt variant
        prompt_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        for episode in self.episodes:
            prompt_variant = episode.get('prompt_variant', 'Unknown')
            # Handle None values
            if prompt_variant is None:
                prompt_variant = 'None'
            prompt_stats[prompt_variant]['total'] += 1
            if episode.get('success', False):
                prompt_stats[prompt_variant]['successful'] += 1
                
        print("\n   By Prompt Variant:")
        for prompt, stats in sorted(prompt_stats.items()):
            rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            print(f"     {prompt}: {stats['successful']}/{stats['total']} ({rate:.1%})")
            
        return {
            'overall': overall_success_rate,
            'by_task': dict(task_stats),
            'by_model': dict(model_stats),
            'by_prompt': dict(prompt_stats)
        }
        
    def analyze_step_patterns(self):
        """Analyze step count and timing patterns."""
        print("\n📈 Step Pattern Analysis")
        
        step_counts = [ep.get('steps_taken', 0) for ep in self.episodes]
        eval_times = [ep.get('evaluation_time', 0) for ep in self.episodes]
        
        if step_counts:
            print(f"   Steps - Min: {min(step_counts)}, Max: {max(step_counts)}, Avg: {sum(step_counts)/len(step_counts):.1f}")
            
        if eval_times:
            print(f"   Time - Min: {min(eval_times):.1f}s, Max: {max(eval_times):.1f}s, Avg: {sum(eval_times)/len(eval_times):.1f}s")
            
        # Steps vs success correlation
        successful_steps = [ep.get('steps_taken', 0) for ep in self.episodes if ep.get('success', False)]
        failed_steps = [ep.get('steps_taken', 0) for ep in self.episodes if not ep.get('success', False)]
        
        if successful_steps and failed_steps:
            avg_successful = sum(successful_steps) / len(successful_steps)
            avg_failed = sum(failed_steps) / len(failed_steps)
            print(f"   Avg Steps - Successful: {avg_successful:.1f}, Failed: {avg_failed:.1f}")
            
    def analyze_action_patterns(self):
        """Analyze action type usage patterns."""
        print("\n🎬 Action Pattern Analysis")
        
        all_action_types = defaultdict(int)
        successful_action_types = defaultdict(int)
        
        for episode in self.episodes:
            actions = episode.get('actions', [])
            is_successful = episode.get('success', False)
            
            episode_action_types = defaultdict(int)
            
            for action_str in actions:
                try:
                    # Extract action JSON from various formats
                    if 'Action:' in action_str:
                        action_json = action_str.split('Action:')[-1].strip()
                    else:
                        action_json = action_str.strip()
                        
                    action_dict = json.loads(action_json)
                    action_type = action_dict.get('action_type', 'unknown')
                    
                    all_action_types[action_type] += 1
                    episode_action_types[action_type] += 1
                    
                    if is_successful:
                        successful_action_types[action_type] += 1
                        
                except Exception:
                    all_action_types['parse_error'] += 1
                    
        print("   Action Type Usage:")
        for action_type, count in sorted(all_action_types.items(), key=lambda x: x[1], reverse=True):
            success_count = successful_action_types.get(action_type, 0)
            success_rate = success_count / count if count > 0 else 0
            print(f"     {action_type}: {count} uses ({success_rate:.1%} in successful episodes)")
            
    def analyze_failure_modes(self):
        """Analyze common failure patterns."""
        print("\n❌ Failure Mode Analysis")
        
        failed_episodes = [ep for ep in self.episodes if not ep.get('success', False)]
        
        if not failed_episodes:
            print("   No failed episodes to analyze")
            return
            
        # Analyze final actions in failed episodes
        final_actions = defaultdict(int)
        infeasible_reasons = []
        
        for episode in failed_episodes:
            actions = episode.get('actions', [])
            if actions:
                try:
                    last_action_str = actions[-1]
                    if 'Action:' in last_action_str:
                        action_json = last_action_str.split('Action:')[-1].strip()
                    else:
                        action_json = last_action_str.strip()
                        
                    action_dict = json.loads(action_json)
                    action_type = action_dict.get('action_type', 'unknown')
                    final_actions[action_type] += 1
                    
                    if action_type == 'status' and action_dict.get('goal_status') == 'infeasible':
                        # Try to extract reason from the response
                        responses = episode.get('responses', [])
                        if responses:
                            last_response = responses[-1]
                            if 'Reason:' in last_response:
                                reason = last_response.split('Reason:')[1].split('Action:')[0].strip()
                                infeasible_reasons.append(reason[:100])  # First 100 chars
                                
                except Exception:
                    final_actions['parse_error'] += 1
                    
        print(f"   Failed Episodes: {len(failed_episodes)}")
        print(f"   Final Actions in Failed Episodes:")
        for action_type, count in sorted(final_actions.items(), key=lambda x: x[1], reverse=True):
            print(f"     {action_type}: {count}")
            
        if infeasible_reasons:
            print(f"\n   Sample Infeasible Reasons:")
            for i, reason in enumerate(infeasible_reasons[:5]):
                print(f"     {i+1}. {reason}...")
                
    def generate_report(self, output_file: Optional[str] = None):
        """Generate a comprehensive analysis report."""
        print("\n📋 Generating Analysis Report")
        
        report = {
            'summary': {
                'total_episodes': len(self.episodes),
                'results_directory': self.results_dir,
                'file_pattern': self.pattern
            }
        }
        
        # Add all analysis results
        report['success_analysis'] = self.analyze_success_rates()
        
        # Create CSV data for detailed analysis
        csv_data = []
        for episode in self.episodes:
            row = {
                'file_name': episode.get('_file_name', ''),
                'task_name': episode.get('task_name', ''),
                'model_name': episode.get('model_name', ''),
                'prompt_variant': episode.get('prompt_variant') or 'None',  # Handle None values
                'success': episode.get('success', False),
                'steps_taken': episode.get('steps_taken', 0),
                'max_steps': episode.get('max_steps', 30),
                'evaluation_time': episode.get('evaluation_time', 0),
                'agent_claimed_done': episode.get('agent_claimed_done', False),
                'task_actually_successful': episode.get('task_actually_successful', False),
            }
            
            # Count action types
            actions = episode.get('actions', [])
            action_counts = defaultdict(int)
            for action_str in actions:
                try:
                    if 'Action:' in action_str:
                        action_json = action_str.split('Action:')[-1].strip()
                    else:
                        action_json = action_str.strip()
                    action_dict = json.loads(action_json)
                    action_type = action_dict.get('action_type', 'unknown')
                    action_counts[action_type] += 1
                except:
                    action_counts['parse_error'] += 1
                    
            # Add most common action types as columns
            for action_type in ['click', 'input_text', 'scroll', 'open_app', 'status', 'navigate_back']:
                row[f'action_{action_type}_count'] = action_counts.get(action_type, 0)
                
            csv_data.append(row)
            
        # Save CSV if pandas is available
        try:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_file = output_file.replace('.json', '.csv') if output_file else os.path.join(self.results_dir, 'batch_analysis.csv')
            df.to_csv(csv_file, index=False)
            print(f"   📊 CSV report saved: {csv_file}")
        except ImportError:
            print("   ⚠️  pandas not available, skipping CSV export")
            
        # Save JSON report
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"   📄 JSON report saved: {output_file}")
            
        return report
        
    def run_full_analysis(self, output_file: Optional[str] = None):
        """Run complete batch analysis."""
        self.load_episodes()
        
        if not self.episodes:
            print("❌ No episodes to analyze")
            return
            
        self.analyze_success_rates()
        self.analyze_step_patterns()
        self.analyze_action_patterns()
        self.analyze_failure_modes()
        
        if output_file:
            self.generate_report(output_file)


def find_episode_files(directory: str, task_filter: Optional[str] = None, 
                      model_filter: Optional[str] = None) -> List[str]:
    """Find episode JSON files with optional filtering.
    
    Args:
        directory: Directory to search
        task_filter: Optional task name filter
        model_filter: Optional model name filter
        
    Returns:
        List of matching file paths
    """
    pattern = "*.json"
    files = glob.glob(os.path.join(directory, pattern))
    
    filtered_files = []
    for file_path in files:
        # Parse filename for basic filtering
        filename = os.path.basename(file_path)
        
        # Skip if doesn't match task filter
        if task_filter and task_filter not in filename:
            continue
            
        # Skip if doesn't match model filter  
        if model_filter and model_filter not in filename:
            continue
            
        filtered_files.append(file_path)
        
    return filtered_files


def main():
    """Main entry point for batch analysis."""
    parser = argparse.ArgumentParser(
        description="Batch Episode Analysis for AndroidWorld Enhanced T3A Agent"
    )
    
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing JSON result files"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json)"
    )
    
    parser.add_argument(
        "--task-filter",
        type=str,
        help="Filter episodes by task name"
    )
    
    parser.add_argument(
        "--model-filter", 
        type=str,
        help="Filter episodes by model name"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for detailed report (JSON format)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis (success rates only)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
        
    print(f"🚀 Batch Episode Analyzer")
    print(f"   Directory: {args.results_dir}")
    print(f"   Pattern: {args.pattern}")
    if args.task_filter:
        print(f"   Task Filter: {args.task_filter}")
    if args.model_filter:
        print(f"   Model Filter: {args.model_filter}")
    print()
    
    # Create analyzer
    analyzer = BatchEpisodeAnalyzer(args.results_dir, args.pattern)
    
    try:
        if args.quick:
            analyzer.load_episodes()
            analyzer.analyze_success_rates()
        else:
            analyzer.run_full_analysis(args.output)
            
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
