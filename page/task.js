(function () {
    'use strict';

    const tasks = window.AUTORLBENCH_TASKS || [];
    const params = new URLSearchParams(window.location.search);
    const taskId = params.get('id');
    const task = tasks.find(item => item.id === taskId);
    const mount = document.getElementById('task-detail-root');

    function escHtml(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function renderMd(text) {
        if (!text) return '';
        const blocks = [];
        text = text.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
            const idx = blocks.length;
            blocks.push(`<pre><code>${escHtml(code.trimEnd())}</code></pre>`);
            return `\x00B${idx}\x00`;
        });
        text = text.replace(/`([^`\n]+)`/g, (_, c) => `<code>${escHtml(c)}</code>`);
        text = text.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        text = text.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        text = text.replace(/^# (.+)$/gm, '<h2>$1</h2>');
        text = text.replace(/((?:^- .+\n?)+)/gm, function (block) {
            const items = block.trim().split('\n').map(l => `<li>${l.replace(/^- /, '')}</li>`).join('');
            return `<ul>${items}</ul>`;
        });
        const lines = text.split('\n');
        const out = [];
        let para = [];
        const structural = s => s.startsWith('<h') || s.startsWith('<ul') || s.startsWith('<pre') || s.startsWith('\x00B');
        for (const line of lines) {
            if (structural(line)) {
                if (para.length) { out.push(`<p>${para.join(' ')}</p>`); para = []; }
                out.push(line);
            } else if (line.trim() === '') {
                if (para.length) { out.push(`<p>${para.join(' ')}</p>`); para = []; }
            } else {
                para.push(line);
            }
        }
        if (para.length) out.push(`<p>${para.join(' ')}</p>`);
        let result = out.join('\n');
        blocks.forEach((b, i) => { result = result.replace(`\x00B${i}\x00`, b); });
        return result;
    }

    function render() {
        if (!mount) return;
        if (!task) {
            mount.innerHTML = '<div class="task-not-found"><h1>Task not found</h1><p>Return to the <a href="index.html#tasks">task catalog</a>.</p></div>';
            return;
        }

        document.title = `${task.title} - AutoRLBench`;
        const img = `task-${task.family.toLowerCase()}.jpeg`;

        const issueContent = task.issueInstruction
            ? renderMd(task.issueInstruction)
            : '<p>Diagnose the run. Identify the root cause using the workspace, configuration, logs, and metrics.</p>';

        const solutionContent = task.solutionInstruction
            ? renderMd(task.solutionInstruction)
            : '<p class="task-eval-coming-soon">Repair-and-rerun evaluation is being prepared.</p>';

        const solutionStatus = task.solutionInstruction ? 'Available' : 'Coming soon';
        const solutionDim = task.solutionInstruction ? '' : ' task-eval-entry--dim';

        mount.innerHTML = `
            <div class="task-breadcrumb"><a href="index.html#tasks">Tasks</a><span>/</span><span>${task.family}</span></div>
            <header class="task-page-header">
                <div class="task-header-tags"><span>${task.family}</span></div>
                <h1>${task.title}</h1>
                <div class="task-header-main">
                    <div class="task-header-left">
                        <p class="task-issue-text">${task.issue}</p>
                        <div class="task-settings">
                            <div><span>Model</span><strong>${task.model}</strong></div>
                            <div><span>Dataset</span><strong>${task.dataset}</strong></div>
                            <div><span>Training</span><strong>${task.framework}</strong></div>
                        </div>
                    </div>
                    <div class="task-header-image">
                        <img src="${img}" alt="${task.family} task illustration" loading="eager">
                    </div>
                </div>
            </header>

            <section class="task-evaluations-section">
                <div class="task-eval-groups">
                    <details class="task-eval-group" open>
                        <summary>
                            <span>Issue</span>
                            <span class="task-eval-group-status task-eval-group-status--live">Available</span>
                        </summary>
                        <div class="task-eval-group-body">
                            ${issueContent}
                        </div>
                    </details>
                    <details class="task-eval-group${task.solutionInstruction ? '' : ' task-eval-group--dim'}">
                        <summary>
                            <span>Solution</span>
                            <span class="task-eval-group-status">${solutionStatus}</span>
                        </summary>
                        <div class="task-eval-group-body">
                            ${solutionContent}
                        </div>
                    </details>
                </div>
            </section>
        `;
    }

    document.addEventListener('DOMContentLoaded', render);
})();
