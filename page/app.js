const TASKS = window.AUTORLBENCH_TASKS || [];

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

// Tasks per domain: the samples in task-data.js. Collection is ongoing.
const DOMAIN_COUNTS = Object.fromEntries(
    TASK_CATEGORIES.map(family => [family, TASKS.filter(task => task.family === family).length])
);

const ISSUE_RESULTS = [
    { model: 'GPT-6-Astra', agent: 'Codex', score: 0.62 },
    { model: 'GPT-6.1-Sol', agent: 'Codex', score: 0.54 },
    { model: 'Claude-Fable-5.1', agent: 'Claude Code', score: 0.77 },
    { model: 'Claude-Opus-5.5', agent: 'Claude Code', score: 0.77 },
    { model: 'GLM-5.3', agent: 'Claude Code', score: 0.92 },
    { model: 'Kimi-K3', agent: 'Claude Code', score: 0.77 },
    { model: 'DeepSeek-V4-Pro', agent: 'Claude Code', score: 0.69 }
];

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
    const heroTasks = document.getElementById('hero-task-count');
    const heroSystems = document.getElementById('hero-system-count');
    if (heroTasks) heroTasks.textContent = String(TASKS.length);
    if (heroSystems) heroSystems.textContent = String(ISSUE_RESULTS.length);

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
        const img = `../assets/task-${task.family.toLowerCase()}.jpeg`;
        return `
            <a href="task.html?id=${encodeURIComponent(task.id)}" class="task-card">
                <div class="task-card-image">
                    <img src="${img}" alt="${task.family} task illustration" loading="lazy">
                    <span class="task-tag">${task.family}</span>
                </div>
                <div class="task-card-body">
                    <h3>${task.title}</h3>
                    <p>${task.brief || task.issue}</p>
                </div>
            </a>
        `;
    }).join('');
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
