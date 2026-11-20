/* Draft prediction comparisons from frozen forecasts and dated outcomes. */
(() => {
  let dataPromise, data, host, observer, evaluations;
  let selected, cardTableData, sortKey = 'actual', sortDirection = -1;
  const predictionPrompt = `the newest mtg set, reality fracture, is about to come out. full set is spoiled. I want you to take a look at all the cards and predict 17lands gih win rates for all of them.
please be thorough and comprehensive. you may look at past sets win rates as well as community reviews and discussions for past sets and current set. your job is to come up with the best predictions possible. I want as output a single table of card and gih win rate. you may spend as much time as you need. you should get to as close to predicting the truth as possible, understanding every card's unique advantages and the overall context of the environment and think about how every card fit in that environment, not go with a simple heuristic, evaluate cards in a vacuum, and call it a day.
please do all your work as needed locally on scratch and only commit your final table as a pr to [https://github.com/wanqizhu/mtg-benchmark](https://github.com/wanqizhu/mtg-benchmark). name your branch/pr with your model name an don't read any other prs. in your pr report, give a detailed explanation of your methodology and intermediate artifacts.`;
  const E = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const F = (value, places = 2) => value == null ? '—' : Number(value).toFixed(places);
  const pct = value => value == null ? '—' : `${F(value)}%`;
  const pretty = value => String(value).replaceAll('_', ' ').replaceAll('-', ' ');
  const link = (url, text) => `<a href="${E(url)}" target="_blank" rel="noopener noreferrer">${E(text)}</a>`;
  const displayName = s => (s.id === 'limitedresources' ? 'LSV + Marshall / LR' : s.name) +
    (s.prediction_group !== 'sandbox' && s.prediction_marker ? ' ' + s.prediction_marker : '');
  const sourceColorClass = s => s.prediction_group === 'sandbox' ? 'dp-source-sandbox' : `dp-source-${s.kind}`;
  const isArchivedSource = s => s.kind === 'ai' && s.prediction_group !== 'sandbox';
  const rankedSources = set => set.sources.filter(s => !isArchivedSource(s));
  const topSource = set => rankedSources(set).sort((a, b) => b.score.pearson - a.score.pearson)[0];
  const source = () => data.sources.find(s => s.id === selected);
  const table = (headers, rows) => `<div class="dp-scroll"><table class="dp-data-table"><thead><tr>${headers.map(h=>`<th>${E(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const scoredCards = () => data.cards.filter(c => c.actual != null && c.games >= 500 && c.values[selected].prediction != null && !c.type.startsWith('Basic Land'));

  const snapshotDate = outcome => new Date(outcome.retrieved_at).toLocaleString('en-US', {
    month:'long', day:'numeric', year:'numeric', hour:'2-digit', minute:'2-digit', hourCycle:'h23', timeZone:'UTC'
  })+' UTC';

  function averageScore(pair) {
    const scores = pair.map(s => s?.score.pearson).filter(Number.isFinite);
    return scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : -Infinity;
  }

  function pairedSources() {
    const { FRA, TRK } = evaluations;
    const trkSources = TRK ? rankedSources(TRK) : [];
    const sourceKey = s => s.kind === 'creator' ? `creator:${s.id}` : `model:${s.name}`;
    const remaining = new Map(trkSources.map(s => [sourceKey(s), s]));
    const rows = rankedSources(FRA).map(s => {
      const key = sourceKey(s);
      const match = remaining.get(key);
      remaining.delete(key);
      return [s, match];
    });
    for (const s of remaining.values()) rows.push([null, s]);
    return rows.sort((a, b) => averageScore(b) - averageScore(a));
  }

  function historicalSourceLinks(url) {
    const tier = url.match(/^https:\/\/(?:www\.)?17lands\.com\/data\/tier_list\/([a-f0-9]+)(?:\?.*)?$/);
    if (!tier) return link(url, 'Original source');
    return `${link('https://www.17lands.com/tier_list/'+tier[1], 'Tier list')} · ${link(url, 'Snapshot JSON')}`;
  }

  function collection(s) {
    let explanation;
    if (s.id === 'draftsim') explanation = `<p class="dp-caution">Timing is provisional for both historical editions and FRA. Hobbit uses secondary snapshots self-dated August 11 and 14; all 188 bot grades agree, but the dates were not independently archived. Strixhaven's current public asset declares April 15 as its latest update; no independent prelaunch copy or edit history was recovered. FRA's collector records September 25 activity but its saved bundle was generated October 6, after launch. These facts do not prove unchanged prelaunch ratings.</p>`;
    else if (s.id === 'limitedresources') explanation = `<p>LSV and Marshall's explicit letter grades come from dated Limited Resources reviews. Hobbit has a complete early transcription. Other sets use partial, manually checked caption reconstruction; ambiguous card attribution and unverified grades are omitted. Qualifying rare reviews are included, sometimes on launch day or within two days afterward. Avatar contributes commons and uncommons only; its rare review missed the three-day cutoff. No video or audio is played on this page.</p>`;
    else explanation = `${s.id === 'casualspike' ? `<p>Casual Spike is a community reviewer who posts as ${link('https://www.reddit.com/user/Casual_Spike/','u/Casual_Spike')} and publishes card-by-card Limited reviews and rating sheets.</p>` : ''}<p>Original native grades and context flags were reconstructed from the earliest qualifying 17Lands tier-list versions. Later edits are excluded. The visual tier-list links open the readable page; Snapshot JSON links identify the dated data used.${s.id === 'alexnikolic' ? " The joint Alex/Marc Hobbit list is excluded from Alex's historical stream." : ''}</p>`;
    return explanation + table(['Set', 'Arena launch', 'Edition used', 'Published cards', 'Coverage'], s.collection.map(r => {
      let deviation = '';
      if (!r.complete) deviation = s.id === 'limitedresources'
        ? `Partial caption transcription; ambiguous card or grade statements omitted.${r.set === 'TLA' ? ' Commons/uncommons only; the rare review was four days after launch.' : ''}`
        : 'The original list does not grade every main-set card.';
      return [E(r.set), E(r.release), E(r.published_or_frozen), E(r.rated_main), `${r.complete ? 'Complete' : 'Partial'}<br>${historicalSourceLinks(r.url)}${deviation ? `<p>${E(deviation)}</p>` : ''}`];
    }));
  }

  function fitDetails(s) {
    const m = s.model;
    const score = m.historical_validation?.rmse_pp;
    let text = `<p>${m.train_n.toLocaleString()} eligible card-set observations across ${m.historical_sets} sets. Each observation receives equal weight. Historical targets are reported PremierDraft FIRST_WEEK GIH for nonbasic main-set cards with at least 500 GIH games.</p><p>Hold out one entire set, remove every matching card name from the training sets, fit on the remaining history, and score the held-out set. Select the model family and penalties by pooled historical validation RMSE; then refit all admitted history. FRA outcomes are inaccessible during fitting. These tuning scores are not nested, unbiased generalization estimates.</p>`;
    if (score != null) text += `<p>Selected historical validation RMSE: <b>${F(score,3)} pp</b>.</p>`;
    if (m.model_type === 'bot_regression') {
      text += `<p>Compare a two-coefficient linear curve with cubic splines containing four through nine coefficients. Spline curvature penalties are 0, 2, 10 or 50; knots are fixed. The selected model has ${m.nominal_degrees_of_freedom} total fitted coefficients and curvature penalty ${m.curvature_lambda}. The input is the published decimal bot rating, not the separate 0–10 written review scale.</p>`;
    } else {
      text += `<p>Compare monotone and unconstrained grade curves and grade/context penalties. Selected: ${m.monotone ? 'monotone' : 'unconstrained'} grade curve; grade penalty ${m.grade_lambda}; context penalty ${m.context_lambda}. Each creator has one shared context-flag coefficient.</p><p>With two explicit base/conditional grades, use only the base-grade bucket and set the context flag to 0. With a conditional-only grade or one grade plus a build-around, synergy or scenario label, use the supplied grade bucket and set the context flag to 1. Multiple labels activate the coefficient only once. A copied identical grade in two fields is one published grade, not a verified pair. Tentative alternative grades do not activate the flag.</p><p>Apply the minimum-ten rule inside each validation fold and the final fit. For a grade with n&lt;10 real memberships, borrow 10−n sample weight from the closest observed native grades' base coefficients. Equal-distance donors share the weight, capped by their actual counts; expand to farther grades if necessary. All adjustments use base coefficients simultaneously. A sparse context coefficient borrows toward zero. Borrowed weight does not create real observations.</p><p>SB and explicit sideboard ratings use F with context flag 0. Published split grades retain their component weights. The published labels remain visible in the card details.</p>`;
    }
    return text;
  }

  function modelDetails(s) {
    const m = s.model;
    if (m.model_type === 'bot_regression') {
      return `<p class="dp-equation">Predicted GIH = Σ coefficient[i] × cubic B-spline basis[i](rating)</p><p>${m.nominal_degrees_of_freedom} fitted coefficients; nominal parameter count ${m.nominal_degrees_of_freedom}. Unconstrained penalized smoother trace ${F(m.unconstrained_smoother_degrees_of_freedom,3)}. Historical rating domain ${m.domain.join('–')}; ratings beyond it use the nearest endpoint.</p>${table(['Spline basis', 'Coefficient (GIH %)'], m.regression_coefficients.map(r => [E(r.term), F(r.coefficient,6)]))}<p>Fixed knot vector: <code>${E(JSON.stringify(m.knots))}</code>. The plotted numeric grade evaluations are not independently fitted coefficients.</p>`;
    }
    const grades = table(['Grade', 'Final GIH %', 'Real n', 'Borrowed weight', 'Donors'], m.grade_mapping.filter(r => r.observed_n > 0).map(r => [E(r.grade), F(r.mapped_wr,5), r.observed_n, F(r.borrowed_weight,1), r.donors.map(d=>`${E(d.grade)}: ${F(d.borrowed_weight,1)}`).join(' · ') || '—']));
    const contexts = m.context_mapping.map(r=>({name:'Shared context flag',value:r.offset_pp,unit:'pp',n:r.observed_n,borrowed:r.borrowed_weight}));
    return `<p class="dp-equation">Predicted GIH = grade coefficient + context flag × shared context coefficient</p><p>Coefficients below are the final historical mappings after minimum-ten smoothing. Real n counts card-set memberships; split grades can contribute to multiple classes. Only grades observed in history are listed. A grade with n=0 has no independently fitted coefficient; its prediction, if needed, is inferred from nearby grades.</p>${grades}${contexts.length ? `<h3>Context coefficient</h3>${table(['Term','Coefficient','Unit','Real n','Borrowed zero weight'],contexts.map(r=>[E(r.name),F(r.value,6),E(r.unit),r.n,r.borrowed]))}` : ''}`;
  }

  function creatorDetails(s) {
    return `<details class="dp-expand"><summary>${E(s.name)} <span>${s.model.historical_sets} sets · ${s.model.train_n.toLocaleString()} historical observations${s.provisional ? ' · provisional dates' : ''}</span></summary><div class="dp-expand-body"><h3>Data collection</h3>${collection(s)}<div class="dp-model-grid"><section><h3>Fit process</h3>${fitDetails(s)}</section><section><h3>Final model</h3>${modelDetails(s)}</section></div></div></details>`;
  }

  function sandboxPromptDetails(fra, trk) {
    const prompt = fra && trk ? fra
      .replace('reality fracture', '[reality fracture (FRA) / star trek (TRK)]')
      .replace('all the new cards and predict', 'all the new cards [TRK only: available in premier draft] and predict')
      : fra || trk;
    const formatted = E(prompt).replace(/\[[^\]\n]*\]/g, phrase => `<span class="dp-prompt-variant">${phrase}</span>`);
    return prompt ? `<details class="dp-expand dp-inner"><summary>Full sandbox prompt</summary><div class="dp-expand-body"><pre class="dp-prompt">${formatted}</pre></div></details>` : '';
  }

  function evaluationDetails(rows) {
    const data=evaluations.FRA;
    const sets=[evaluations.FRA,evaluations.TRK].filter(Boolean);
    const archived = data.sources.filter(isArchivedSource);
    const submissions = rows.filter(([fra, trk]) => (fra ?? trk).kind === 'ai');
    const submissionLink = s => s ? link(s.url, s.post_release ? 'Submission (post-release)' : 'Submission') : '—';
    return `<details class="dp-expand"><summary>Methodology</summary><div class="dp-expand-body">
      <p>GIH win rates taken on ${sets.map(set=>`${E(set.expansion)}: ${link(set.outcome.url,snapshotDate(set.outcome))}`).join(' · ')}.</p>
      <section class="dp-method-section" aria-labelledby="dp-sandbox-methodology">
        <h3 id="dp-sandbox-methodology" class="dp-method-heading dp-source-sandbox">Sandbox model predictions</h3>
        <p>For both Reality Fracture and Star Trek, these models used only the supplied cards, rules, and historical data. They had no internet access to read current-set win rates or other reviews.</p>
        ${sandboxPromptDetails(data.sandbox_prompt, evaluations.TRK?.sandbox_prompt)}
        ${data.sandbox_audit ? `<p>${link(data.sandbox_audit.transcript_pr,'FRA transcript PR for all models')}</p>` : ''}
        <details class="dp-expand dp-inner"><summary>Sandbox model submissions</summary><div class="dp-expand-body">${table(['Model', 'FRA', 'TRK'], submissions.map(([fra, trk])=>[E((fra ?? trk).name), submissionLink(fra), submissionLink(trk)]))}<p>${link('https://github.com/wanqizhu/mtg-win-rate-eval/blob/main/README.md','Star Trek sandbox dataset and rules')}</p></div></details>
      </section>
      <section class="dp-method-section" aria-labelledby="dp-creator-methodology">
      <h3 id="dp-creator-methodology" class="dp-method-heading dp-source-creator">Content creator models</h3>
      <p>I pulled historical tier lists for up to five earlier sets and compared them against realized win rates. I then used this data to fit a model converting categorical predictions (e.g. A−, 3.5) to win rates.</p>
      <p>After aggregating the historical data, we fit a regression from each letter-grade bucket to its realized win rate, with smoothing between nearby tiers. Buckets with fewer than 10 observations receive enough weight from nearby buckets to reach 10. Each letter-grade creator model has one shared context coefficient: explicit base/conditional pairs use only the base bucket; conditional-only grades and single grades with context labels use their grade bucket plus the context coefficient. Draftsim's decimal ratings use a monotone spline with five fitted coefficients. Model settings are selected by holding out earlier sets; the final models are fitted on historical data and applied to FRA and TRK without refitting.</p>
      <p>This is my best effort representation of content creator ratings. LLMs have the advantage of more granular win rate predictions; although the content creator models have the advantage of seeing realized win rates for their buckets in earlier sets, the lack of granularity speaks unfairly to content creators' ability to discern cards. I'm sure they have nonzero predictability among cards rated in the same bucket, but chose coarser buckets for reader comprehension.</p>
      <details class="dp-expand dp-inner" id="dp-current-rating-sources"><summary>Current-set rating sources</summary><div class="dp-expand-body">
        ${table(['Set','Creator','Original rating sources'],sets.flatMap(set=>set.sources.filter(s=>s.kind==='creator').map(s=>[E(set.expansion),E(s.name),(s.current_sources || [{url:s.url,label:'Current-set ratings'}]).map(r=>link(r.url,r.label)).join('<br>')+(s.provenance?`<details><summary>Edition used</summary>${E(s.provenance)}</details>`:'')])))}
      </div></details>
      </section>
      <section class="dp-method-section" aria-labelledby="dp-original-methodology">
        <h3 id="dp-original-methodology" class="dp-method-heading dp-source-ai">Original model predictions (Archived)</h3>
        <p>These runs are archived because it’s hard to separate the influence of public tier lists from the models’ own predictions.</p>
        <p>Original model predictions were made for Reality Fracture before set release, using the same prompt. These predictions allowed models to use content creator reviews available before the set’s September 29, 2026 MTG Arena release.</p>
        <details class="dp-expand dp-inner"><summary>Full prompt</summary><div class="dp-expand-body"><pre class="dp-prompt">${E(predictionPrompt)}</pre></div></details>
        <details class="dp-expand dp-inner"><summary>Original model submissions</summary><div class="dp-expand-body">${table(['Model','FRA Pearson r','Original prediction source'],archived.map(s=>[E(s.name),F(s.score.pearson,3),link(s.url,'Prediction PR')]))}</div></details>
      </section>
    </div></details>`;
  }

  function scoreCell(s, set) {
    if (!s) return '<td>—</td>';
    return `<td><button class="dp-score ${sourceColorClass(s)}" data-source="${E(s.id)}" data-set="${set}" aria-pressed="${data.expansion===set&&selected===s.id}" title="${E(s.name)} · ${set} · ${s.score.n} scored cards">${F(s.score.pearson,3)}</button></td>`;
  }

  function cardTableMarkup() {
    return `<details class="dp-expand" id="dp-card-table-details">
      <summary>All card predictions</summary>
      <div class="dp-expand-body">
        <div class="dp-table-controls">
          ${evaluations.TRK ? `<div class="dp-table-toggle" role="group" aria-label="Card prediction set">
            <button type="button" data-card-set="FRA">FRA</button>
            <button type="button" data-card-set="TRK">TRK</button>
          </div>` : ''}
          <input class="dp-search" id="dp-table-search" type="search" aria-label="Search card table" placeholder="Search cards…">
          <span id="dp-table-count"></span>
          <a class="link-button" id="dp-download" download>Download all predictions CSV</a>
        </div>
        <div class="dp-full-table">
          <table class="dp-data-table" aria-label="All card predictions">
            <thead id="dp-card-columns"></thead>
            <tbody id="dp-card-rows"></tbody>
          </table>
        </div>
      </div>
    </details>`;
  }

  function setCardTableData(set) {
    if (set !== cardTableData) {
      cardTableData = set;
      const showOutcomes = set.card_outcomes_available !== false;
      sortKey = showOutcomes ? 'actual' : 'name';
      sortDirection = showOutcomes ? -1 : 1;
      const csvName = set.expansion === 'TRK' ? (showOutcomes ? 'draft-predictions-trk' : 'draft-predictions-trk-summary')
        : evaluations.fraRefreshed ? 'draft-predictions-fra-current' : 'draft-predictions';
      host.querySelector('#dp-download').href = `data/${csvName}.csv`;
      host.querySelectorAll('[data-card-set]').forEach(button => {
        button.setAttribute('aria-pressed', String(button.dataset.cardSet === set.expansion));
      });
      host.querySelector('#dp-card-columns').innerHTML = `<tr>
        <th><button class="dp-sort" data-sort="name">Card ↕</button></th>
        ${showOutcomes ? '<th class="dp-numeric"><button class="dp-sort" data-sort="actual">Observed<br>GIH ↕</button></th>' : ''}
        ${set.sources.map(s => `<th class="dp-numeric"><button class="dp-sort ${sourceColorClass(s)}" data-sort="${E(s.id)}">${E(displayName(s))} ↕</button></th>`).join('')}
      </tr>`;
      host.querySelector('.dp-full-table').scrollTo(0, 0);
    }
    renderCardTable();
  }

  function skeleton() {
    const rows = pairedSources();
    host.innerHTML = `<div class="draft-page">
      <div class="dp-heading">
        <div><h1>Win Rate Predictions</h1><p>How good are people and LLMs at predicting card win rates in a new format?</p></div>
        <div class="dp-meta">FRA &amp; TRK · Premier Draft GIH win rate from ${link('https://www.17lands.com/', '17Lands')}</div>
      </div>
      <div class="dp-overview">
        <section class="dp-board">
          <h2>Overall comparison</h2>
          <p class="dp-comparison-pool">Ranked by average Pearson r across available sets. Click a score for card predictions.</p>
          <div class="dp-source-legend"><span class="dp-source-sandbox">Sandbox models</span><span class="dp-source-creator">Content creators</span></div>
          <table aria-label="Overall prediction scores">
            <thead><tr><th>Prediction source</th><th>FRA</th><th>Star Trek</th></tr></thead>
            <tbody>${rows.map(([fra, trk]) => {
              const s = fra ?? trk;
              return `<tr><td><span class="dp-source ${sourceColorClass(s)}">${E(displayName(s))}</span>${fra && trk && fra.name !== trk.name ? `<small class="dp-card-meta">TRK: ${E(trk.name)}</small>` : ''}</td>${scoreCell(fra, 'FRA')}${scoreCell(trk, 'TRK')}</tr>`;
            }).join('')}</tbody>
          </table>
        </section>
        <section class="dp-chart-card">
          <div class="dp-chart-header"><div><h2 id="dp-selected-name"></h2><p id="dp-chart-description"></p></div></div>
          <div id="dp-plot" class="dp-plot"></div>
          <div class="dp-legend">${[['Common', 'blue'], ['Uncommon', 'green'], ['Rare', 'gold'], ['Mythic', 'red']].map(([rarity, color]) => `<span><i class="dp-dot" style="background:var(--dp-${color})"></i>${rarity}</span>`).join('')}<span>Hover for rates · click for card details</span></div>
        </section>
      </div>
      <div class="dp-extensions">
        ${evaluationDetails(rows)}
        <details class="dp-expand"><summary>Content Creator Historical Data Details</summary><div class="dp-expand-body dp-creator-group">${evaluations.FRA.sources.filter(s => s.kind === 'creator').map(creatorDetails).join('')}</div></details>
        ${cardTableMarkup()}
      </div>
      <div id="dp-tooltip" class="dp-tooltip" role="tooltip" hidden></div>
      <dialog id="dp-card-dialog" class="dp-dialog" aria-labelledby="dp-dialog-title"><div id="dp-dialog-content"></div></dialog>
    </div>`;
    host.querySelectorAll('[data-source]').forEach(button => button.addEventListener('click', () => {
      data = evaluations[button.dataset.set];
      selected = button.dataset.source;
      renderPlot();
      setCardTableData(data);
    }));
    host.querySelectorAll('[data-card-set]').forEach(button => button.addEventListener('click', () => {
      setCardTableData(evaluations[button.dataset.cardSet]);
    }));
    host.querySelector('#dp-table-search').addEventListener('input', renderCardTable);
    host.querySelector('#dp-card-columns').addEventListener('click', event => {
      const button = event.target.closest('[data-sort]');
      if (!button) return;
      const key = button.dataset.sort;
      sortDirection = key === sortKey ? -sortDirection : key === 'name' ? 1 : -1;
      sortKey = key;
      renderCardTable();
    });
    host.querySelector('#dp-card-dialog').addEventListener('click', event => {
      if (event.target.closest('[data-close]')) host.querySelector('#dp-card-dialog').close();
    });
    setCardTableData(data);
    renderPlot();
    observer = new ResizeObserver(() => { if (host?.querySelector('#dp-plot')) renderPlot(); });
    observer.observe(host.querySelector('#dp-plot'));
  }

  function ticks(min, max, count=5) {
    const raw=(max-min)/count, power=10**Math.floor(Math.log10(raw));
    const step=[1,2,5,10].find(x=>x*power>=raw)*power;
    const result=[];for(let x=Math.ceil(min/step)*step;x<=max+step*.001;x+=step)result.push(Math.abs(x)<1e-8?0:Number(x.toPrecision(8)));
    return result;
  }

  function renderPlot() {
    const s=source(), plot=host.querySelector('#dp-plot');
    host.querySelector('#dp-selected-name').textContent=displayName(s)+' · '+data.name;
    host.querySelector('#dp-selected-name').className=sourceColorClass(s);
    const showOutcomes = data.card_outcomes_available !== false;
    host.querySelector('#dp-chart-description').textContent = `${s.score.n} scored cards` +
      (showOutcomes ? ' · predicted vs observed · dashed line = perfect prediction' : '');
    host.querySelectorAll('[data-source]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.source===selected&&b.dataset.set===data.expansion)));
    hideTooltip();
    host.querySelector('.dp-legend').hidden = !showOutcomes;
    if (!showOutcomes) {
      plot.innerHTML = `<p>See ${link(data.outcome.url, 'Star Trek card win rates on 17Lands')}.</p>`;
      return;
    }
    const cards=scoredCards(), width=Math.max(270,plot.clientWidth), height=360, left=58,right=14,top=15,bottom=47;
    let xs=cards.map(c=>c.values[selected].prediction);
    let ys=cards.map(c=>c.actual);
    let xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
    xmin=ymin=Math.min(xmin,ymin);xmax=ymax=Math.max(xmax,ymax);
    const xp=Math.max(1,(xmax-xmin)*.075);xmin-=xp;xmax+=xp;
    ymin-=xp;ymax+=xp;
    const x=v=>left+(v-xmin)/(xmax-xmin)*(width-left-right),y=v=>top+(ymax-v)/(ymax-ymin)*(height-top-bottom);
    const xt=ticks(xmin,xmax,width<420?3:5),yt=ticks(ymin,ymax,5);
    const grid=xt.map(v=>`<line class="dp-grid" x1="${x(v)}" y1="${top}" x2="${x(v)}" y2="${height-bottom}"/><text class="dp-tick" x="${x(v)}" y="${height-bottom+18}" text-anchor="middle">${v}</text>`).join('')+yt.map(v=>`<line class="dp-grid" x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}"/><text class="dp-tick" x="${left-10}" y="${y(v)+4}" text-anchor="end">${v}</text>`).join('');
    const reference=`<line class="dp-reference" x1="${x(xmin)}" y1="${y(xmin)}" x2="${x(xmax)}" y2="${y(xmax)}"/>`;
    const colors={common:'blue',uncommon:'green',rare:'gold',mythic:'red'};
    plot.innerHTML=`<svg role="group" aria-label="${E(displayName(s))}: ${cards.length} card predictions versus observed GIH" viewBox="0 0 ${width} ${height}"><title>${E(displayName(s))} card-level predictions</title>${grid}${reference}${cards.map((c,i)=>`<circle class="dp-point" data-card="${data.cards.indexOf(c)}" cx="${x(xs[i])}" cy="${y(ys[i])}" r="4.2" fill="var(--dp-${colors[c.rarity]||'blue'})" tabindex="0" role="button" aria-label="${E(c.name)}; predicted ${F(c.values[selected].prediction)} percent; observed ${F(c.actual)} percent"/>`).join('')}<text class="dp-axis" x="${(left+width-right)/2}" y="${height-7}" text-anchor="middle">Predicted GIH (%)</text><text class="dp-axis" x="15" y="${(top+height-bottom)/2}" text-anchor="middle" transform="rotate(-90 15 ${(top+height-bottom)/2})">Observed GIH (%)</text></svg>`;
    plot.querySelectorAll('[data-card]').forEach(point=>{
      const card=data.cards[Number(point.dataset.card)];
      point.addEventListener('pointerenter',event=>showTooltip(card,event.clientX,event.clientY));
      point.addEventListener('pointermove',event=>showTooltip(card,event.clientX,event.clientY));
      point.addEventListener('pointerleave',hideTooltip);
      point.addEventListener('focus',()=>{const r=point.getBoundingClientRect();showTooltip(card,r.right,r.top);});
      point.addEventListener('blur',hideTooltip);
      point.addEventListener('click',()=>showCard(card));
      point.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();showCard(card);}});
    });
  }

  function showTooltip(card, clientX, clientY) {
    const tip=host.querySelector('#dp-tooltip'), v=card.values[selected], error=v.prediction-card.actual;
    tip.innerHTML=`<b>${E(card.name)}</b><div class="dp-muted">${E(card.cost)} · ${E(card.rarity)} · ${E(card.color)}</div><div class="dp-tooltip-rates"><span>Predicted GIH</span><span>${pct(v.prediction)}</span><span>Observed GIH</span><span>${pct(card.actual)}</span><span>Difference</span><span>${error>=0?'+':''}${F(error)} pp</span></div>${v.grade!=null?`<div>Published grade: ${E(v.grade)}</div>`:''}<div class="dp-muted">${card.games.toLocaleString()} GIH games · click for image and stats</div>`;
    tip.hidden=false;tip.style.left=`${Math.max(8,Math.min(clientX+14,innerWidth-276))}px`;tip.style.top=`${Math.max(8,Math.min(clientY+14,innerHeight-tip.offsetHeight-12))}px`;
  }
  function hideTooltip(){const tip=host?.querySelector('#dp-tooltip');if(tip)tip.hidden=true;}
  function showCard(card, s = source()) {
    hideTooltip();
    const v = card.values[s.id];
    const error = v.prediction == null || card.actual == null ? null : v.prediction - card.actual;
    const adjustment = s.model?.model_type === 'native_bucket' && v.prediction != null
      ? (v.applied_contexts?.length ? s.model.context_mapping[0].offset_pp : 0) : null;
    host.querySelector('#dp-dialog-content').innerHTML=`<div class="dp-dialog-heading"><h2 id="dp-dialog-title">${E(card.name)}</h2><button data-close aria-label="Close card details">×</button></div>
      <div class="dp-card-detail-layout"><figure class="dp-card-images">${card.images.map(i=>`<img class="dp-card-image" src="${E(i.url)}" alt="${E(i.name)} — Scryfall card image" width="488" height="680" referrerpolicy="no-referrer">`).join('')}<p class="dp-image-status" hidden>Image unavailable. ${link(card.url,'View on Scryfall')}</p><figcaption>${link(card.url,'Image via Scryfall')}</figcaption></figure>
      <section class="dp-card-stats" aria-label="Card prediction stats"><p class="dp-modal-source ${sourceColorClass(s)}">${E(displayName(s))}</p><div class="dp-detail-rates"><div><small>Predicted GIH</small><strong>${pct(v.prediction)}</strong></div>${card.games == null ? '' : `<div><small>Observed GIH</small><strong>${pct(card.actual)}</strong></div><div><small>Difference</small><strong>${error==null?'—':`${error>=0?'+':''}${F(error)} pp`}</strong></div><div><small>GIH games</small><strong>${card.games.toLocaleString()}</strong></div>`}</div>${v.grade!=null?`<p><b>Published grade:</b> ${E(v.grade)}${v.contexts?.length?`<br><b>Published context:</b> ${E(v.contexts.map(pretty).join(', '))}`:''}${adjustment!=null?`<br><b>Grade used:</b> ${E(v.grade_used)}<br><b>Context adjustment:</b> ${adjustment>0?'+':''}${F(adjustment)} pp`:''}${v.source_url?`<br>${link(v.source_url,'Original rating source')}`:''}</p>`:''}</section></div>`;
    host.querySelectorAll('.dp-card-image').forEach(img=>img.addEventListener('error',()=>{img.hidden=true;host.querySelector('.dp-image-status').hidden=false;}));
    host.querySelector('#dp-card-dialog').showModal();
  }

  function renderCardTable() {
    const selectedSource = data === cardTableData ? source() : topSource(cardTableData);
    const { cards, sources, catalog_cards: totalCards } = cardTableData;
    const showOutcomes = cardTableData.card_outcomes_available !== false;
    const query=host.querySelector('#dp-table-search').value.trim().toLowerCase();
    const rows=cards.filter(c=>c.name.toLowerCase().includes(query));
    const value=c=>sortKey==='name'?c.name:sortKey==='actual'?c.actual:c.values[sortKey].prediction;
    rows.sort((a,b)=>{const aa=value(a),bb=value(b);if(aa==null)return bb==null?a.name.localeCompare(b.name):1;if(bb==null)return -1;return sortDirection*(typeof aa==='string'?aa.localeCompare(bb):aa-bb)||a.name.localeCompare(b.name);});
    host.querySelectorAll('[data-sort]').forEach(b=>{b.closest('th').setAttribute('aria-sort',b.dataset.sort===sortKey?(sortDirection===1?'ascending':'descending'):'none');});
    host.querySelector('#dp-table-count').textContent=`${rows.length} of ${totalCards} cards`;
    host.querySelector('#dp-card-rows').innerHTML=rows.map(c=>`<tr><td><button class="dp-card-button" data-table-card="${cards.indexOf(c)}">${E(c.name)}</button><span class="dp-card-meta">${E(c.set.toUpperCase())} · ${E(c.rarity)}${showOutcomes&&(c.actual==null||c.values[selectedSource.id].prediction==null)?' · unscored':''}</span></td>${showOutcomes?`<td class="dp-numeric">${pct(c.actual)}</td>`:''}${sources.map(s=>`<td class="dp-numeric">${pct(c.values[s.id].prediction)}${s.kind==='creator'&&c.values[s.id].grade!=null?`<span class="dp-cell-grade">${E(c.values[s.id].grade)}</span>`:''}</td>`).join('')}</tr>`).join('');
    host.querySelectorAll('[data-table-card]').forEach(b=>b.addEventListener('click',()=>showCard(cards[Number(b.dataset.tableCard)], selectedSource)));
  }
  function dispose(){observer?.disconnect();observer=null;hideTooltip();}
  async function mount(container,{isCurrent=()=>true}={}) {
    dispose();host=container;container.innerHTML='<p class="muted">Loading draft predictions…</p>';
    const load=async (path, optional=false)=>{
      const response=await fetch(`data/${path}.json`,{cache:'no-cache'});
      if (optional && response.status===404) return null;
      if (!response.ok) throw new Error('Could not load draft prediction data');
      const set=await response.json();
      if (set.catalog_cards!==set.cards.length || !set.sources.length || new Set(set.sources.map(s=>s.id)).size!==set.sources.length || set.cards.some(c=>set.sources.some(s=>!c.values[s.id]))) throw new Error('Unexpected draft prediction dataset');
      return set;
    };
    dataPromise ??= Promise.all([load('draft-predictions-fra-current',true),
      load('draft-predictions-trk',true).then(local => local ?? load('draft-predictions-trk-summary',true))])
      .then(async ([current,trek])=>({FRA:current??await load('draft-predictions'),TRK:trek,fraRefreshed:Boolean(current)}))
      .catch(error=>{dataPromise=null;throw error;});
    const loaded=await dataPromise;
    if (!isCurrent()) return;
    evaluations=loaded;
    data = loaded.FRA;
    selected = topSource(data).id;
    cardTableData = null;
    const url=new URL(location.href);
    if (url.searchParams.has('draft')) {url.searchParams.delete('draft');history.replaceState(history.state,'',url);}
    skeleton();
  }
  window.DraftPredictions={mount,dispose};
})();
