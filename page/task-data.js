window.RSIBENCH_RL_TASKS = [
    {
        id: 'slime_retool_length_hack',
        title: 'Length bonus reward hack',
        family: 'Reward',
        model: 'Qwen3-4B',
        dataset: 'DAPO-Math-17k',
        framework: 'slime / GRPO',
        issue: 'Training Qwen3-4B with GRPO on DAPO-Math-17k. The model hacks the reward by generating longer answers.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'reward_design/task_function_call',
        title: 'TACO function-call scoring',
        family: 'Reward',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'TACO',
        framework: 'verl / GRPO',
        issue: 'The function-call reward path is removed.',
        evaluations: ['issue']
    },
    {
        id: 'ppo_critic/aime25_single_sample_variance',
        title: 'AIME25 single-sample variance',
        family: 'Configuration',
        model: 'Qwen3-4B-Base',
        dataset: 'AIME25',
        framework: 'verl / PPO-Critic evaluation',
        issue: 'Evaluating Qwen3-4B-Base on AIME25. One sample per question makes the score too noisy to judge the run.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'rlve_hyper/task_008',
        title: 'Learning rate',
        family: 'Configuration',
        model: 'Qwen2.5-0.5B-Instruct',
        dataset: 'RLVE held-out environments',
        framework: 'RLVE / slime / GRPO',
        issue: 'The learning rate is set to 1e-9.',
        evaluations: ['issue']
    },
    {
        id: 'rlve_hyper/task_012',
        title: 'Hidden dropout',
        family: 'Configuration',
        model: 'Qwen2.5-0.5B-Instruct',
        dataset: 'RLVE held-out environments',
        framework: 'RLVE / slime / GRPO',
        issue: 'Hidden dropout is set to 0.5.',
        evaluations: ['issue']
    },
    {
        id: 'rlve_hyper/task_019',
        title: 'Learning-rate schedule',
        family: 'Configuration',
        model: 'Qwen2.5-0.5B-Instruct',
        dataset: 'RLVE held-out environments',
        framework: 'RLVE / slime / GRPO',
        issue: 'Warmup is longer than the observed training run.',
        evaluations: ['issue']
    },
    {
        id: 'verl_hyper/qwen_dapo_hparam/task_hparam_overlong_023',
        title: 'DAPO response length',
        family: 'Configuration',
        model: 'Qwen2.5-0.5B-Instruct',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / GRPO',
        issue: 'The response length is capped at 256 tokens.',
        evaluations: ['issue']
    },
    {
        id: 'task_train_too_hard',
        title: 'Training data is too hard',
        family: 'Data',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'TACO',
        framework: 'verl / GRPO',
        issue: 'Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training set is very hard while the validation set is easy.',
        evaluations: ['issue']
    },
    {
        id: 'verl_implementation/llama_dapo_implementation/task_monkey_patch_011',
        title: 'Missing flash-attention hook',
        family: 'Infrastructure',
        model: 'Llama-3.2-1B-Instruct',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / GRPO',
        issue: 'Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. The baseline flash-attention hook is never installed.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'verl_implementation/verl_reward_implementation/task_engine_base_025',
        title: 'VERL base engine',
        family: 'Infrastructure',
        model: 'Qwen2.5-0.5B-Instruct',
        dataset: 'GSM8K',
        framework: 'verl / GRPO',
        issue: 'Core BaseEngine methods are overwritten.',
        evaluations: ['issue']
    },
    {
        id: 'search_r1_zero_query_vector',
        title: 'Zero query vector in retrieval',
        family: 'Environment',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'HotpotQA distractor',
        framework: 'Search-R1 / verl / GRPO',
        issue: 'Training Qwen2.5-1.5B-Instruct with GRPO on HotpotQA. The query vector stays all zeros, so every query retrieves the same documents.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'ptdbench_rank_distillation_collapse',
        title: 'Ranking distillation collapse',
        family: 'Algorithm',
        model: 'DeepSeek-R1-Distill-Qwen-1.5B',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / ranking distillation',
        issue: 'Training a 1.5B student with on-policy ranking distillation on DAPO-Math-17k. The ranking loss improves while generation quality collapses.',
        evaluations: ['issue']
    }
];
