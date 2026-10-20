#!/usr/bin/env node
/* Extract evaluation instructions from tasks/ into task-data.js */
const fs = require('fs');
const path = require('path');

const tasksDir = path.join(__dirname, '..', 'tasks');
const taskDataPath = path.join(__dirname, 'task-data.js');

const TASK_IDS = [
    'configuration/verl_aime25_single_sample_protocol',
    'reward/slime_retool_length_reward_bonus',
    'data/verl_taco_difficulty_mismatch',
    'environment/search_r1_zero_query_embedding',
    'algorithm/verl_grpo_uid_after_repeat',
    'infrastructure/verl_missing_flash_attn_patch'
];

const tasks = [];

for (const id of TASK_IDS) {
    const taskPath = path.join(tasksDir, id);
    const readmePath = path.join(taskPath, 'README.md');

    if (!fs.existsSync(readmePath)) {
        console.warn(`WARNING: ${id} README.md not found`);
        continue;
    }

    const readme = fs.readFileSync(readmePath, 'utf-8');

    // Extract title, family, model, dataset, framework from existing task-data.js structure
    // For now, read the instruction files
    const issueInstructionPath = path.join(taskPath, 'evaluations/issue/instruction.md');
    const solutionInstructionPath = path.join(taskPath, 'evaluations/solution/instruction.md');

    let issueInstruction = null;
    let solutionInstruction = null;

    if (fs.existsSync(issueInstructionPath)) {
        issueInstruction = fs.readFileSync(issueInstructionPath, 'utf-8').trim();
    }

    if (fs.existsSync(solutionInstructionPath)) {
        solutionInstruction = fs.readFileSync(solutionInstructionPath, 'utf-8').trim();
    }

    const evaluations = [];
    if (issueInstruction) evaluations.push('issue');
    if (solutionInstruction) evaluations.push('solution');

    tasks.push({
        id,
        issueInstruction,
        solutionInstruction,
        evaluations
    });
}

console.log(JSON.stringify(tasks, null, 2));
