const TASKS = window.RSIBENCH_RL_TASKS || [];

const TASK_CATEGORIES = [
    'Configuration',
    'Reward',
    'Data',
    'Environment',
    'Algorithm',
    'Infrastructure'
];

const DOMAIN_BLURBS = {
    Configuration: 'Bad hyperparameters and training settings.',
    Reward: 'Rewards that miss or punish the right behavior.',
    Data: 'Input, format, or distribution / difficulty issues.',
    Environment: 'Unstable or mis-specified training environments.',
    Algorithm: 'Algorithm design flaws, unstable updates, and exploration failures.',
    Infrastructure: 'Runtime, engine, and systems failures.'
};

const DOMAIN_COUNTS = {
    Configuration: 6,
    Reward: 7,
    Data: 1,
    Environment: 1,
    Algorithm: 1,
    Infrastructure: 4
};

const RESULT_TASK_IDS = [
    'verl_hyper/qwen_dapo_hparam/task_hparam_overlong_023',
    'verl_implementation/data_format/task_agent_loop_022-llama-dapo-math',
    'verl_implementation/llama_dapo_implementation/task_monkey_patch_011',
    'verl_implementation/verl_reward_implementation/task_engine_base_025',
    'rlve_hyper/task_001',
    'rlve_hyper/task_008',
    'rlve_hyper/task_012',
    'rlve_hyper/task_019',
    'reward_design/task_evaluator',
    'reward_design/task_function_call'
];

const ISSUE_RESULTS = [
    { model: 'GPT-6-Astra', agent: 'Codex', score: 0.62 },
    { model: 'GPT-6.1-Sol', agent: 'Codex', score: 0.54 },
    { model: 'Claude-Fable-5.1', agent: 'Claude Code', score: 0.77 },
    { model: 'Claude-Opus-5.5', agent: 'Claude Code', score: 0.77 },
    { model: 'GLM-5.3', agent: 'Claude Code', score: 0.92 },
    { model: 'Kimi-K3', agent: 'Claude Code', score: 0.77 },
    { model: 'DeepSeek-V4-Pro', agent: 'Claude Code', score: 0.69 }
];

let activeResultTask = 'all';

function tasksForFamily(family) {
    return TASKS.filter(task => task.family === family);
}

function renderDomainCards() {
    const mount = document.getElementById('domain-cards');
    if (!mount) return;

    const domainCount = document.getElementById('domain-count');
    const activeCount = document.getElementById('active-task-count');
    if (domainCount) domainCount.textContent = String(TASK_CATEGORIES.length);
    if (activeCount) activeCount.textContent = String(Object.values(DOMAIN_COUNTS).reduce((sum, count) => sum + count, 0));

    mount.innerHTML = TASK_CATEGORIES.map((family, index) => {
        const count = DOMAIN_COUNTS[family] || 0;
        return `
            <div class="domain-card">
                <span class="domain-card-index">${String(index + 1).padStart(2, '0')}</span>
                <span class="domain-card-name">${family}</span>
                <span class="domain-card-blurb">${DOMAIN_BLURBS[family] || ''}</span>
                <span class="domain-card-count">${count}</span>
            </div>
        `;
    }).join('');
}

function renderTaskCatalog() {
    const mount = document.getElementById('sample-cards');
    if (!mount) return;
    mount.innerHTML = TASK_CATEGORIES.map(family => {
        const task = tasksForFamily(family)[0];
        if (!task) {
            return `
                <div class="task-card is-empty">
                    <span class="task-tag">${family}</span>
                    <h3>Sample coming soon</h3>
                </div>
            `;
        }
        return `
            <a href="task.html?id=${encodeURIComponent(task.id)}" class="task-card">
                <span class="task-tag">${task.family}</span>
                <h3>${task.title}</h3>
                <p>${task.issue}</p>
                <div class="task-meta">
                    <span>${task.model}</span>
                    <span>${task.framework}</span>
                </div>
            </a>
        `;
    }).join('');
}

function resultTaskLabel(taskId) {
    return TASKS.find(task => task.id === taskId)?.title || taskId;
}

function average(scores) {
    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

function renderLeaderboard() {
    const mount = document.querySelector('.leaderboard-list');
    if (!mount) return;
    const rows = ISSUE_RESULTS
        .map((result, index) => ({ ...result, index }))
        .sort((a, b) => b.score - a.score || a.index - b.index);
    mount.innerHTML = rows.map(result => `
            <div class="lb-row">
                <div class="lb-model-cell"><span class="lb-model-name">${result.model}</span></div>
                <div class="lb-agent-cell"><span class="lb-agent-name">${result.agent}</span></div>
                <div class="lb-scores">
                    <div class="lb-score-item">
                        <div class="lb-score-bar"><div class="lb-score-fill" style="width:${result.score * 100}%"></div></div>
                    </div>
                    <div class="lb-issue-score">${result.score.toFixed(2)}</div>
                </div>
            </div>
        `).join('');
}

function initLeaderboard() {}

function initScrollMotion() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    document.documentElement.classList.add('motion-ready');
    const sections = document.querySelectorAll('body > section:not(.hero), body > footer');
    sections.forEach(section => section.classList.add('scroll-reveal'));
    const revealAll = () => sections.forEach(section => section.classList.add('is-visible'));
    requestAnimationFrame(() => {
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            if (rect.top < window.innerHeight * 1.15 && rect.bottom > 0) section.classList.add('is-visible');
        });
    });
    if (!('IntersectionObserver' in window)) return revealAll();
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
        });
    }, { threshold: 0.08, rootMargin: '0px 0px -8% 0px' });
    sections.forEach(section => observer.observe(section));
    window.setTimeout(revealAll, 1200);
}

document.addEventListener('DOMContentLoaded', () => {
    renderDomainCards();
    renderTaskCatalog();
    renderLeaderboard();
    initLeaderboard();
    initScrollMotion();
});
