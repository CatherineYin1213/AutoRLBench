(function () {
    'use strict';

    const tasks = window.RSIBENCH_RL_TASKS || [];
    const params = new URLSearchParams(window.location.search);
    const task = tasks.find(item => item.id === params.get('id'));
    const mount = document.getElementById('task-detail-root');

    function render() {
        if (!mount) return;
        if (!task) {
            mount.innerHTML = '<div class="task-not-found"><h1>Task not found</h1><p>Return to the <a href="index.html#tasks">task catalog</a>.</p></div>';
            return;
        }

        document.title = `${task.title} - AutoRLBench`;
        mount.innerHTML = `
            <div class="task-breadcrumb"><a href="index.html#tasks">Tasks</a><span>/</span><span>${task.family}</span></div>
            <header class="task-page-header">
                <div class="task-header-tags"><span>${task.family}</span><span>Issue evaluation</span></div>
                <h1>${task.title}</h1>
                <p>${task.issue}</p>
                <div class="task-settings" aria-label="Run settings">
                    <div><span>Model</span><strong>${task.model}</strong></div>
                    <div><span>Dataset</span><strong>${task.dataset}</strong></div>
                    <div><span>Training</span><strong>${task.framework}</strong></div>
                </div>
            </header>

            <section class="task-evaluation-section">
                <div class="mode-section-heading">
                    <div><span class="task-section-kicker">Available now</span><h2>Issue Evaluation</h2></div>
                </div>
                <div class="issue-summary-panel">
                    <strong>Diagnose the run.</strong>
                    <p>Identify the root cause using the workspace, configuration, logs, and metrics.</p>
                </div>
            </section>

            <section class="task-evaluation-section">
                <div class="coming-soon-block">
                    <span>COMING SOON</span>
                    <div><h2>Solution</h2><p>Repair-and-rerun evaluation is being prepared.</p></div>
                </div>
            </section>
        `;
    }

    document.addEventListener('DOMContentLoaded', render);
})();
