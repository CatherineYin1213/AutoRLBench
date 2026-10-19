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
        issue: 'Evaluating Qwen3-4B-Base on AIME25. One sample per question makes the score too noisy to judge the run.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'reward/slime_retool_length_reward_bonus',
        title: 'Length bonus reward hack',
        family: 'Reward',
        model: 'Qwen3-4B',
        dataset: 'DAPO-Math-17k',
        framework: 'slime / GRPO',
        issue: 'Training Qwen3-4B with GRPO on DAPO-Math-17k. The model hacks the reward by generating longer answers.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'data/verl_taco_difficulty_mismatch',
        title: 'Training data is too hard',
        family: 'Data',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'TACO',
        framework: 'verl / GRPO',
        issue: 'Training Qwen2.5-1.5B-Instruct with GRPO on TACO. The training set is very hard while the validation set is easy.',
        evaluations: ['issue']
    },
    {
        id: 'environment/search_r1_zero_query_embedding',
        title: 'Zero query vector in retrieval',
        family: 'Environment',
        model: 'Qwen2.5-1.5B-Instruct',
        dataset: 'HotpotQA distractor',
        framework: 'Search-R1 / verl / GRPO',
        issue: 'Training Qwen2.5-1.5B-Instruct with GRPO on HotpotQA. The query vector stays all zeros, so every query retrieves the same documents.',
        evaluations: ['issue', 'solution']
    },
    {
        id: 'algorithm/rank_distillation_collapse',
        title: 'Ranking distillation collapse',
        family: 'Algorithm',
        model: 'DeepSeek-R1-Distill-Qwen-1.5B',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / ranking distillation',
        issue: 'Training a 1.5B student with on-policy ranking distillation on DAPO-Math-17k. The ranking loss improves while generation quality collapses.',
        evaluations: ['issue']
    },
    {
        id: 'infrastructure/verl_missing_flash_attn_patch',
        title: 'Missing flash-attention hook',
        family: 'Infrastructure',
        model: 'Llama-3.2-1B-Instruct',
        dataset: 'DAPO-Math-17k',
        framework: 'verl / GRPO',
        issue: 'Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. The baseline flash-attention hook is never installed.',
        evaluations: ['issue', 'solution']
    }
];
