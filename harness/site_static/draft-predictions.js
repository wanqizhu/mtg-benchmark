/* Frozen FRA comparison: original AI forecasts plus historical creator fits. */
(() => {
  let dataPromise, data, host, observer;
  let selected, sortKey = 'actual', sortDirection = -1;
  const predictionPrompt = `the newest mtg set, reality fracture, is about to come out. full set is spoiled. I want you to take a look at all the cards and predict 17lands gih win rates for all of them.
please be thorough and comprehensive. you may look at past sets win rates as well as community reviews and discussions for past sets and current set. your job is to come up with the best predictions possible. I want as output a single table of card and gih win rate. you may spend as much time as you need. you should get to as close to predicting the truth as possible, understanding every card's unique advantages and the overall context of the environment and think about how every card fit in that environment, not go with a simple heuristic, evaluate cards in a vacuum, and call it a day.
please do all your work as needed locally on scratch and only commit your final table as a pr to [https://github.com/wanqizhu/mtg-benchmark](https://github.com/wanqizhu/mtg-benchmark). name your branch/pr with your model name an don't read any other prs. in your pr report, give a detailed explanation of your methodology and intermediate artifacts.`;
  const E = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const F = (value, places = 2) => value == null ? '—' : Number(value).toFixed(places);
  const pct = value => value == null ? '—' : `${F(value)}%`;
  const pretty = value => String(value).replaceAll('_', ' ').replaceAll('-', ' ');
  const link = (url, text) => `<a href="${E(url)}" target="_blank" rel="noopener noreferrer">${E(text)}</a>`;
  const displayName = s => s.id === 'limitedresources' ? 'LSV + Marshall / LR' : s.name;
  const source = () => data.sources.find(s => s.id === selected);
  const table = (headers, rows) => `<div class="dp-scroll"><table class="dp-data-table"><thead><tr>${headers.map(h=>`<th>${E(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const sharedCards = () => data.cards.filter(c => c.shared);

  function collection(s) {
    let explanation;
    if (s.id === 'draftsim') explanation = `<p class="dp-caution">Timing is provisional for both historical editions and FRA. Hobbit uses secondary snapshots self-dated August 11 and 14; all 188 bot grades agree, but the dates were not independently archived. Strixhaven's current public asset declares April 15 as its latest update; no independent prelaunch copy or edit history was recovered. FRA's collector records September 25 activity but its saved bundle was generated October 6, after launch. These facts do not prove unchanged prelaunch ratings.</p>`;
    else if (s.id === 'limitedresources') explanation = `<p>LSV and Marshall's explicit letter grades come from dated Limited Resources reviews. Hobbit has a complete early transcription. Other sets use partial, manually checked caption reconstruction; ambiguous card attribution and unverified grades are omitted. Qualifying rare reviews are included, sometimes on launch day or within two days afterward. Avatar contributes commons and uncommons only; its rare review missed the three-day cutoff. No video or audio is played on this page.</p>`;
    else explanation = `<p>Original native grades and context flags were reconstructed from the earliest qualifying 17Lands tier-list versions. Later edits are excluded.${s.id === 'alexnikolic' ? " The joint Alex/Marc Hobbit list is excluded from Alex's historical stream." : ''}</p>`;
    return explanation + table(['Set', 'Arena launch', 'Edition used', 'Published cards', 'Coverage'], s.collection.map(r => {
      let deviation = '';
      if (!r.complete) deviation = s.id === 'limitedresources'
        ? `Partial caption transcription; ambiguous card or grade statements omitted.${r.set === 'TLA' ? ' Commons/uncommons only; the rare review was four days after launch.' : ''}`
        : 'The original list does not grade every main-set card.';
      return [E(r.set), E(r.release), E(r.published_or_frozen), E(r.rated_main), `${r.complete ? 'Complete' : 'Partial'}<br>${link(r.url, 'Original source')}${deviation ? `<p>${E(deviation)}</p>` : ''}`];
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
      text += `<p>Compare monotone and unconstrained grade curves, grade and context penalties, and eligible conditional-grade-gap effects. Selected: ${m.monotone ? 'monotone' : 'unconstrained'} grade curve; grade penalty ${m.grade_lambda}; context penalty ${m.context_lambda}; conditional gaps ${m.use_gaps ? 'enabled' : 'disabled'}.</p><p>Apply the minimum-ten rule inside each validation fold and the final fit. For a grade with n&lt;10 real memberships, borrow 10−n sample weight from the closest observed native grades' base coefficients. Equal-distance donors share the weight, capped by their actual counts; expand to farther grades if necessary. All adjustments use base coefficients simultaneously. Sparse context effects borrow toward a zero-effect ordinary-card baseline. Borrowed weight does not create real observations.</p><p>SB and explicit sideboard ratings use F. Published split grades retain their component weights. Ordinary grades supply the input when available; explicit conditional-only grades retain their context. The published labels remain visible in the card details.</p>`;
    }
    return text;
  }

  function modelDetails(s) {
    const m = s.model;
    if (m.model_type === 'bot_regression') {
      return `<p class="dp-equation">Predicted GIH = Σ coefficient[i] × cubic B-spline basis[i](rating)</p><p>${m.nominal_degrees_of_freedom} fitted coefficients; nominal parameter count ${m.nominal_degrees_of_freedom}. Unconstrained penalized smoother trace ${F(m.unconstrained_smoother_degrees_of_freedom,3)}. Historical rating domain ${m.domain.join('–')}; ratings beyond it use the nearest endpoint.</p>${table(['Spline basis', 'Coefficient (GIH %)'], m.regression_coefficients.map(r => [E(r.term), F(r.coefficient,6)]))}<p>Fixed knot vector: <code>${E(JSON.stringify(m.knots))}</code>. The plotted numeric grade evaluations are not independently fitted coefficients.</p>`;
    }
    const grades = table(['Grade', 'Final GIH %', 'Real n', 'Borrowed weight', 'Donors'], m.grade_mapping.filter(r => r.observed_n > 0).map(r => [E(r.grade), F(r.mapped_wr,5), r.observed_n, F(r.borrowed_weight,1), r.donors.map(d=>`${E(d.grade)}: ${F(d.borrowed_weight,1)}`).join(' · ') || '—']));
    const contexts = [...m.context_mapping.map(r=>({name:r.bucket+' offset',value:r.offset_pp,unit:'pp',n:r.observed_n,borrowed:r.borrowed_weight})), ...m.context_gap_mapping.map(r=>({name:r.bucket+' conditional-grade gap',value:r.pp_per_native_grade_step,unit:'pp / native step',n:r.observed_n,borrowed:r.borrowed_weight}))];
    return `<p class="dp-equation">Predicted GIH = grade coefficient + context offsets + selected grade-gap effects</p><p>Coefficients below are the final historical mappings after minimum-ten smoothing. Real n counts card-set memberships; split grades can contribute to multiple classes. Only grades observed in history are listed. A grade with n=0 has no independently fitted coefficient; its prediction, if needed, is inferred from nearby grades.</p>${grades}${contexts.length ? `<h3>Context terms</h3>${table(['Term','Coefficient','Unit','Real n','Borrowed zero weight'],contexts.map(r=>[E(pretty(r.name)),F(r.value,6),E(r.unit),r.n,r.borrowed]))}` : ''}`;
  }

  function creatorDetails(s) {
    return `<details class="dp-expand"><summary>${E(s.name)} <span>${s.model.historical_sets} sets · ${s.model.train_n.toLocaleString()} historical observations${s.provisional ? ' · provisional dates' : ''}</span></summary><div class="dp-expand-body"><h3>Data collection</h3>${collection(s)}<div class="dp-model-grid"><section><h3>Fit process</h3>${fitDetails(s)}</section><section><h3>Final model</h3>${modelDetails(s)}</section></div></div></details>`;
  }

  function evaluationDetails() {
    const ais = data.sources.filter(s=>s.kind==='ai');
    return `<details class="dp-expand"><summary>Methodology</summary><div class="dp-expand-body">
      <p>Overall results use the ${data.shared_cards} FRA main-set cards, GIH win rate taken on ${link(data.outcome.url,'October 7, 2026 at 05:58 UTC')}.</p>
      <h3>Model predictions</h3>
      <p>Model predictions were made for every card before set release, using the same prompt.</p>
      <details class="dp-expand dp-inner"><summary>Full prompt</summary><div class="dp-expand-body"><pre class="dp-prompt">${E(predictionPrompt)}</pre></div></details>
      <details class="dp-expand dp-inner"><summary>Original model submissions</summary><div class="dp-expand-body">${table(['Model','Original prediction source'],ais.map(s=>[E(s.name),link(s.url,'Prediction PR')]))}</div></details>
      <h3>Content creator models</h3>
      <p>I pulled historical tier lists for up to five earlier sets and compared them against realized win rates. I then used this data to fit a model converting categorical predictions (e.g. A−, 3.5) to win rates.</p>
      <p>After aggregating the historical data, we fit a monotone regression from each letter-grade bucket to its realized win rate, with smoothing between nearby tiers. Buckets with fewer than 10 observations receive enough weight from nearby buckets to reach 10. Draftsim's decimal ratings use a monotone spline with five fitted coefficients. Model settings are selected by holding out earlier sets; the final models are fitted on historical data and applied to FRA.</p>
      <p>I believe this is a fair representation of predicted card strength. LLMs have the advantage of more granular win rate predictions; the content creator models have the advantage of seeing realized win rates for their buckets in earlier sets. Smoothing and monotonicity constraints help limit overfitting.</p>
    </div></details>`;
  }

  function skeleton() {
    const ordered = [...data.sources].sort((a,b)=>b.score.pearson-a.score.pearson);
    selected = selected && data.sources.some(s=>s.id===selected) ? selected : ordered[0].id;
    host.innerHTML = `<div class="draft-page"><div class="dp-heading"><div><h1>Win rate predictions: Reality Fractured</h1><p>Model predictions made before set release; content creator ratings converted to win rates based on a historical model, see details below.</p></div><div class="dp-meta">FRA · Premier Draft GIH win rate</div></div><div class="dp-overview"><section class="dp-board"><h2>Overall comparison</h2><div class="dp-source-legend"><span class="dp-source-ai">Models</span><span class="dp-source-creator">Content creators</span></div><table aria-label="Overall prediction scores"><thead><tr><th>Prediction source</th><th>Pearson r</th><th>RMSE<br>(pp)</th></tr></thead><tbody>${ordered.map(s=>`<tr data-source-row="${E(s.id)}"><td><button class="dp-source dp-source-${s.kind}" data-source="${E(s.id)}" aria-pressed="false">${E(displayName(s))}</button></td><td>${F(s.score.pearson,3)}</td><td>${F(s.score.rmse)}</td></tr>`).join('')}</tbody></table></section><section class="dp-chart-card"><div class="dp-chart-header"><div><h2 id="dp-selected-name"></h2><p id="dp-chart-description"></p></div></div><div id="dp-plot" class="dp-plot"></div><div class="dp-legend">${[['common','blue'],['uncommon','green'],['rare','gold'],['mythic','red']].map(([r,c])=>`<span><i class="dp-dot" style="background:var(--dp-${c})"></i>${r[0].toUpperCase()+r.slice(1)}</span>`).join('')}<span>Hover for rates · click for card details</span></div></section></div><div class="dp-extensions">${evaluationDetails()}<details class="dp-expand"><summary>Content Creator Historical Data Details</summary><div class="dp-expand-body dp-creator-group">${data.sources.filter(s=>s.kind==='creator').map(creatorDetails).join('')}</div></details><details class="dp-expand" id="dp-card-table-details"><summary>All card predictions <span>${data.catalog_cards} cards · all ten sources · published creator grades</span></summary><div class="dp-expand-body"><div class="dp-table-controls"><input class="dp-search" id="dp-table-search" type="search" aria-label="Search card table" placeholder="Search cards…"><span id="dp-table-count"></span><a class="link-button" id="dp-download" href="data/draft-predictions.csv" download>Download all predictions CSV</a></div><div class="dp-full-table"><table class="dp-data-table" aria-label="All card predictions"><thead><tr><th><button class="dp-sort" data-sort="name">Card ↕</button></th><th class="dp-numeric"><button class="dp-sort" data-sort="actual">Observed<br>GIH ↕</button></th>${data.sources.map(s=>`<th class="dp-numeric"><button class="dp-sort" data-sort="${E(s.id)}">${E(s.name)} ↕</button></th>`).join('')}</tr></thead><tbody id="dp-card-rows"></tbody></table></div></div></details></div><div id="dp-tooltip" class="dp-tooltip" role="tooltip" hidden></div><dialog id="dp-card-dialog" class="dp-dialog" aria-labelledby="dp-dialog-title"><div id="dp-dialog-content"></div></dialog></div>`;
    host.querySelectorAll('[data-source]').forEach(button=>button.addEventListener('click',()=>{selected=button.dataset.source;renderPlot();}));
    host.querySelector('#dp-table-search').addEventListener('input',renderCardTable);
    host.querySelectorAll('[data-sort]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.sort;sortDirection=key===sortKey?-sortDirection:key==='name'?1:-1;sortKey=key;renderCardTable();}));
    host.querySelector('#dp-card-dialog').addEventListener('click',event=>{if(event.target.closest('[data-close]'))host.querySelector('#dp-card-dialog').close();});
    renderCardTable();renderPlot();
    observer = new ResizeObserver(()=>{if(host?.querySelector('#dp-plot'))renderPlot();});
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
    host.querySelector('#dp-selected-name').textContent=s.name;
    host.querySelector('#dp-chart-description').textContent='Predicted vs observed · dashed line = perfect prediction';
    host.querySelectorAll('[data-source-row]').forEach(row=>row.classList.toggle('dp-selected',row.dataset.sourceRow===selected));
    host.querySelectorAll('[data-source]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.source===selected)));
    hideTooltip();
    const cards=sharedCards(), width=Math.max(270,plot.clientWidth), height=360, left=58,right=14,top=15,bottom=47;
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
    plot.innerHTML=`<svg role="group" aria-label="${E(s.name)}: ${cards.length} card predictions versus observed GIH" viewBox="0 0 ${width} ${height}"><title>${E(s.name)} card-level predictions</title>${grid}${reference}${cards.map((c,i)=>`<circle class="dp-point" data-card="${data.cards.indexOf(c)}" cx="${x(xs[i])}" cy="${y(ys[i])}" r="4.2" fill="var(--dp-${colors[c.rarity]||'blue'})" tabindex="0" role="button" aria-label="${E(c.name)}; predicted ${F(c.values[selected].prediction)} percent; observed ${F(c.actual)} percent"/>`).join('')}<text class="dp-axis" x="${(left+width-right)/2}" y="${height-7}" text-anchor="middle">Predicted GIH (%)</text><text class="dp-axis" x="15" y="${(top+height-bottom)/2}" text-anchor="middle" transform="rotate(-90 15 ${(top+height-bottom)/2})">Observed GIH (%)</text></svg>`;
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
  function showCard(card) {
    hideTooltip();const s=source(),v=card.values[selected],error=v.prediction==null||card.actual==null?null:v.prediction-card.actual;
    host.querySelector('#dp-dialog-content').innerHTML=`<div class="dp-dialog-heading"><h2 id="dp-dialog-title">${E(card.name)}</h2><button data-close aria-label="Close card details">×</button></div>
      <div class="dp-card-detail-layout"><figure class="dp-card-images">${card.images.map(i=>`<img class="dp-card-image" src="${E(i.url)}" alt="${E(i.name)} — Scryfall card image" width="488" height="680" referrerpolicy="no-referrer">`).join('')}<p class="dp-image-status" hidden>Image unavailable. ${link(card.url,'View on Scryfall')}</p><figcaption>${link(card.url,'Image via Scryfall')}</figcaption></figure>
      <section class="dp-card-stats" aria-label="Card prediction stats"><p class="dp-modal-source dp-source-${s.kind}">${E(displayName(s))}</p><div class="dp-detail-rates"><div><small>Predicted GIH</small><strong>${pct(v.prediction)}</strong></div><div><small>Observed GIH</small><strong>${pct(card.actual)}</strong></div><div><small>Difference</small><strong>${error==null?'—':`${error>=0?'+':''}${F(error)} pp`}</strong></div><div><small>GIH games</small><strong>${card.games.toLocaleString()}</strong></div></div>${v.grade!=null?`<p><b>Published grade:</b> ${E(v.grade)}${v.contexts?.length?`<br><b>Context:</b> ${E(v.contexts.map(pretty).join(', '))}`:''}</p>`:''}</section></div>`;
    host.querySelectorAll('.dp-card-image').forEach(img=>img.addEventListener('error',()=>{img.hidden=true;host.querySelector('.dp-image-status').hidden=false;}));
    host.querySelector('#dp-card-dialog').showModal();
  }

  function renderCardTable() {
    const query=host.querySelector('#dp-table-search').value.trim().toLowerCase();
    const rows=data.cards.filter(c=>c.name.toLowerCase().includes(query));
    const value=c=>sortKey==='name'?c.name:sortKey==='actual'?c.actual:c.values[sortKey].prediction;
    rows.sort((a,b)=>{const aa=value(a),bb=value(b);if(aa==null)return bb==null?a.name.localeCompare(b.name):1;if(bb==null)return -1;return sortDirection*(typeof aa==='string'?aa.localeCompare(bb):aa-bb)||a.name.localeCompare(b.name);});
    host.querySelectorAll('[data-sort]').forEach(b=>{b.closest('th').setAttribute('aria-sort',b.dataset.sort===sortKey?(sortDirection===1?'ascending':'descending'):'none');});
    host.querySelector('#dp-table-count').textContent=`${rows.length} of ${data.catalog_cards} cards`;
    host.querySelector('#dp-card-rows').innerHTML=rows.map(c=>`<tr><td><button class="dp-card-button" data-table-card="${data.cards.indexOf(c)}">${E(c.name)}</button><span class="dp-card-meta">${E(c.set.toUpperCase())} · ${E(c.rarity)}${c.shared?'':' · unscored'}</span></td><td class="dp-numeric">${pct(c.actual)}</td>${data.sources.map(s=>`<td class="dp-numeric">${pct(c.values[s.id].prediction)}${s.kind==='creator'&&c.values[s.id].grade!=null?`<span class="dp-cell-grade">${E(c.values[s.id].grade)}</span>`:''}</td>`).join('')}</tr>`).join('');
    host.querySelectorAll('[data-table-card]').forEach(b=>b.addEventListener('click',()=>showCard(data.cards[Number(b.dataset.tableCard)])));
  }
  function dispose(){observer?.disconnect();observer=null;hideTooltip();}
  async function mount(container,{isCurrent=()=>true}={}) {
    dispose();host=container;container.innerHTML='<p class="muted">Loading draft predictions…</p>';
    dataPromise ??= fetch('data/draft-predictions.json?v=2').then(r=>{if(!r.ok)throw new Error('Could not load draft prediction data');return r.json();}).catch(error=>{dataPromise=null;throw error;});
    const loaded=await dataPromise;if(!isCurrent())return;data=loaded;
    if(data.sources.length!==10||data.catalog_cards!==295)throw new Error('Unexpected draft prediction dataset');
    skeleton();
  }
  window.DraftPredictions={mount,dispose};
})();
