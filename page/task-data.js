/*
 * Public task index. Each id is the task path under tasks/ in the repository.
 */
window.AUTORLBENCH_TASKS = [
    {
        id: 'configuration/verl_aime25_single_sample_protocol',
        title: 'AIME25 single-sample variance',
        family: 'Configuration',
        model: 'Qwen3-4B-Base',
        dataset: 'AIME25',
        framework: 'verl / PPO-Critic evaluation',
        brief: 'One sample per question makes evaluation too noisy.',
        issue: 'Evaluating Qwen3-4B-Base on AIME25. One sample per question makes the score too noisy to judge the run.',
        issueInstruction: '# RL evaluation result diagnosis\n\nReview the staged post-training evaluation and determine whether the observed AIME25 result is sufficient evidence of a post-training failure, or whether the result itself is unreliable. Treat the workspace as a historical snapshot: inspect its source, the evaluation configuration and protocol, the evaluator code, and the frozen evaluation evidence.\n\nConsider how many questions the benchmark has, how generation is sampled, and how many samples per question the protocol draws. A single number is not by itself proof of any single cause.\n\nDo not launch a full training run, use the network, or modify the historical snapshot. The benchmark, model, and generation settings are fixed.\n\n## Deliverable\n\nWrite one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the verdict, identify the concrete root cause, and explain the causal chain using specific workspace evidence. Distinguish root causes from symptoms.',
        solutionInstruction: '# Repair the evaluation protocol\n\nReview the staged AIME25 evaluation and determine whether the single-run result is reliable. If the evaluation protocol is defective, repair it in the staged workspace at `/workspace`.\n\nThe repaired default protocol must:\n\n- keep the 30-question AIME25 benchmark\n- keep stochastic generation at temperature `1.0` and top-p `1.0`\n- use `32` samples per question\n- aggregate repeated samples with the `mean` reducer\n\nConcretely, update the workspace evaluation configuration (`config/eval_config.yaml`) that `run_eval.py` consumes so that `run_eval.py --protocol-only` reports `samples_per_question: 32` and `expected_generations: 960`.\n\nDo not modify the preserved original evaluation evidence. Do not use the network.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'reward/slime_retool_length_reward_bonus',
        title: 'Length bonus reward hack',
        family: 'Reward',
        model: 'Qwen3-4B',
        dataset: 'DAPO-Math-17k',
        framework: 'slime / GRPO',
        brief: 'Model hacks the reward by generating longer answers.',
        issue: 'Training Qwen3-4B with GRPO on DAPO-Math-17k. The model hacks the reward by generating longer answers.',
        issueInstruction: '# RL training run diagnosis\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a historical snapshot: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence.\n\nUse held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals. Training reward alone does not establish held-out health.\n\nDo not launch a full training run, use the network, or modify the historical snapshot.\n\n## Deliverable\n\nWrite one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the verdict, identify the concrete root cause or causes, and explain the causal chain using specific workspace evidence.',
        solutionInstruction: '# Improve the staged experiment\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a snapshot you may repair: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence.\n\nIf you find a defect, fix it in the staged workspace at `/workspace`. Then run the experiment as a foreground process and wait until it returns. Do not use the network. Do not use `nohup`, `&`, `tmux`, `screen`, or any detach mechanism.\n\nThe launch script is `examples/retool/retool_qwen3_4b_rl.sh`. The effective run must use the HF checkpoint and training data at `/workspace`, with `--num-rollout 70`, `--eval-interval 10`, offline W&B logging to `$AUTORLBENCH_RESULTS/experiment_logs`.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'data/verl_taco_difficulty_mismatch',
        title: 'Training data is too hard',
        family: 'Data',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'TACO',
        framework: 'verl / GRPO',
        brief: 'Training set is very hard while validation set is easy.',
        issue: 'Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training set is very hard while the validation set is easy.',
        issueInstruction: '# RL training run diagnosis\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a historical snapshot: inspect its source, effective configuration, launch scripts, data, reward logic, and any logs or metrics that are present.\n\nUse held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals. Training reward alone does not establish held-out health.\n\nDo not launch a full training run, use the network, or modify the historical snapshot.\n\n## Deliverable\n\nWrite one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the verdict, identify the concrete root cause or causes, and explain the causal chain using specific workspace evidence.',
        evaluations: ['issue']
    },
    {
        id: 'environment/search_r1_zero_query_embedding',
        title: 'Zero query vector in retrieval',
        family: 'Environment',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'HotpotQA distractor',
        framework: 'Search-R1 / verl / GRPO',
        brief: 'Query vector stays all zeros, retrieves same documents.',
        issue: 'Training Qwen2.5-1.5B-Instruct with GRPO on HotpotQA. The query vector stays all zeros, so every query retrieves the same documents.',
        issueInstruction: '# RL training run diagnosis\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a historical snapshot: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence.\n\nUse held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals.\n\nDo not launch a full training run, use the network, or modify the historical snapshot.\n\n## Deliverable\n\nWrite one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the verdict, identify the concrete root cause or causes, and explain the causal chain using specific workspace evidence.',
        solutionInstruction: '# Improve the staged experiment\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. If you find a defect, fix it in the staged workspace at `/workspace`. Then run the experiment as a foreground process and wait until it returns.\n\nThe launch script is `run.py`. Run it from `/workspace`:\n\n```bash\npython3 run.py --run-name graded --runs-dir "$AUTORLBENCH_RESULTS/experiment_logs"\n```\n\nThis entry point starts the local retrieval server, streams training output, and shuts the server down when training ends. Do not start the retriever yourself.\n\nThe effective run must use base model, train/validation data, retrieval corpus and query encoder at `/workspace/shared/`, with budget in `config.json`, offline W&B logging. Run `python3 run.py --check` first to validate the workspace.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'algorithm/verl_grpo_uid_after_repeat',
        title: 'GRPO uid assigned after rollout repeat',
        family: 'Algorithm',
        model: 'Llama-3.2-1B-Instruct',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / GRPO',
        brief: 'GRPO silently degraded to baseline-free REINFORCE.',
        issue: 'Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. The uid assignment was moved after the rollout repeat step, making every GRPO group a singleton and silently degrading training to baseline-free REINFORCE.',
        issueInstruction: '# RL training run diagnosis\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a historical snapshot: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence.\n\nUse held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals.\n\nDo not launch a full training run, use the network, or modify the historical snapshot.\n\n## Deliverable\n\nWrite one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the verdict, identify the concrete root cause or causes, and explain the causal chain using specific workspace evidence.',
        evaluations: ['issue']
    },
    {
        id: 'infrastructure/verl_missing_flash_attn_patch',
        title: 'Missing flash-attention hook',
        family: 'Infrastructure',
        model: 'Llama-3.2-1B-Instruct',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / GRPO',
        brief: 'Flash-attention hook is never installed.',
        issue: 'Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. The baseline flash-attention hook is never installed.',
        issueInstruction: '# RL training run diagnosis\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. Treat the workspace as a historical snapshot: inspect its source, effective configuration, launch scripts, data processing, reward logic, logs, tracked metrics, and frozen evaluation evidence.\n\nUse held-out evaluation evidence together with rollout, truncation, runtime, optimization, and training signals.\n\nDo not launch a full training run, use the network, or modify the historical snapshot.\n\n## Deliverable\n\nWrite one concise UTF-8 diagnosis to `$AUTORLBENCH_RESULTS/agent_answer.md`. State the verdict, identify the concrete root cause or causes, and explain the causal chain using specific workspace evidence.',
        solutionInstruction: '# Improve the staged experiment\n\nReview the staged post-training experiment and determine whether it is healthy or unhealthy. If you find a defect, fix it in the staged workspace at `/workspace`. Then run the experiment as a foreground process and wait until it returns. Do not use the network or detach mechanisms.\n\nThe launch script is `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh`. The effective run must use model, train/test data at `/workspace`, with offline W&B logging to `$AUTORLBENCH_RESULTS/experiment_logs`. Keep the training-step budget and evaluation frequency at their defaults.',
        evaluations: ['issue', 'solution']
    }
];
