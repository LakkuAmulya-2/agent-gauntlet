"""
Agent Gauntlet — Generation Sampler

Guide Section 15: "inspect actual generations during training"
Guide Section 8: "sample outputs frequently and inspect them"

Run this periodically during training to check for:
- Reward hacking (rising reward but bad behavior)
- Repeated action loops
- Invalid JSON format
- Trivial completions
- Suspicious shortcuts

Usage:
    python scripts/sample_generations.py --model-dir outputs/gauntlet-easy-20260425 --url http://localhost:8000
    python scripts/sample_generations.py --model-dir outputs/gauntlet-easy-20260425 --n-episodes 5
"""

from __future__ import annotations

import argparse
import json
from typing import List

from agent_gauntlet import AgentAction, AgentGauntletEnv
from agent_gauntlet.models import ActionType


def sample_with_model(model_dir: str, env_url: str, n_episodes: int, difficulty: str):
    """Sample generations from a trained model and inspect them."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("transformers not installed. Run: pip install transformers torch")
        return

    from train_grpo import SYSTEM_PROMPT

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    with AgentGauntletEnv(base_url=env_url).sync() as env:
        for ep in range(n_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep+1}/{n_episodes}")
            print(f"{'='*60}")

            result = env.reset(difficulty=difficulty)
            obs = result.observation
            print(f"Task: {obs.task_description[:100]}...")
            print(f"Tools: {obs.available_tools}")

            ep_reward = 0.0
            actions_taken = []
            suspicious = []

            for step in range(obs.max_steps):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"TASK: {obs.task_description}\n"
                        f"TOOLS: {', '.join(obs.available_tools)}\n"
                        f"Step {obs.current_step}/{obs.max_steps} | "
                        f"Budget: {obs.api_calls_made}/{obs.api_calls_budget}"
                    )}
                ]

                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=False, enable_thinking=False
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=200,
                        temperature=0.7, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                try:
                    action_data = json.loads(generated.strip())
                    action = AgentAction(
                        action_type=action_data.get("action_type", "call_tool"),
                        tool_name=action_data.get("tool_name"),
                        reasoning=action_data.get("reasoning", ""),
                        failure_detected=action_data.get("failure_detected"),
                        recovery_strategy=action_data.get("recovery_strategy"),
                        task_result=action_data.get("task_result"),
                    )
                    valid_json = True
                except json.JSONDecodeError:
                    print(f"  Step {step}: INVALID JSON: {generated[:100]}")
                    suspicious.append(f"step {step}: invalid JSON")
                    action = AgentAction(
                        action_type=ActionType.CALL_TOOL.value,
                        tool_name=obs.available_tools[0] if obs.available_tools else "unknown",
                        reasoning="fallback after JSON parse error",
                    )
                    valid_json = False

                result = env.step(action)
                ep_reward += result.reward
                actions_taken.append(action.action_type)

                if len(actions_taken) >= 3 and len(set(actions_taken[-3:])) == 1:
                    suspicious.append(f"step {step}: 3 identical actions in a row ({actions_taken[-1]})")

                if action.action_type == ActionType.COMPLETE_TASK.value and step < 3:
                    suspicious.append(f"step {step}: trivial completion at step {step}")

                print(f"  Step {step}: {action.action_type}"
                      + (f" ({action.tool_name})" if action.tool_name else "")
                      + f" -> reward={result.reward:.3f}"
                      + (" ok" if valid_json else " INVALID"))

                obs = result.observation
                if obs.is_done:
                    print(f"  Done: {obs.termination_reason}")
                    break

            print(f"\nEpisode summary:")
            print(f"  Total reward: {ep_reward:.4f}")
            print(f"  Actions: {actions_taken}")
            if suspicious:
                print(f"  SUSPICIOUS PATTERNS:")
                for s in suspicious:
                    print(f"     - {s}")
            else:
                print(f"  No suspicious patterns detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to trained model")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--difficulty", default="easy")
    args = parser.parse_args()

    sample_with_model(args.model_dir, args.url, args.n_episodes, args.difficulty)


if __name__ == "__main__":
    main()
