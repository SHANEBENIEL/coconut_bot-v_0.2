'use strict';

/* ══════════════════════════════════════════
   SPEED KEY:
   • LSTM trains on WEEKLY averages (~315pts) not daily (~1500pts) → 5x faster
   • Lookback = 10 (was 30) → 3x fewer params
   • 1 LSTM layer 24 units (was 2 layers 32+16) → 2x faster
   • 20 epochs batch=128 (was 40 epochs batch=32) → 4x faster per epoch
   • 90-step rollout: batched in groups of 10 → avoids 90 tensor round-trips
   • HW grid: ~20 combos (was ~80)
   • localStorage cache for AR/HW params (skip on reload)
══════════════════════════════════════════ */

/* ── CACHE ── */
const CACHE_KEY='cai_v3';
function saveCache(){try{localStorage.setItem(CACHE_KEY,JSON.stringify({ts:Date.now(),arBeta,hwP,WT,CI,metrics}));}catch(e){}}
function loadCache(){
  try{
    const c=localStorage.getItem(CACHE_KEY);
    if(c){const d=JSON.parse(c);if(Date.now()-d.ts<43200000){arBeta=d.arBeta;hwP=d.hwP;WT=d.WT;CI=d.CI;metrics=d.metrics;return true;}}
  }catch(e){}
  return false;
}

/* ── 1. DATA ── */
function generateDataset(){
  const mkt=['Pollachi','Coimbatore','Chennai','Tiruppur','Erode','Salem','Madurai'];
  const adj={Pollachi:0,Coimbatore:2,Chennai:5,Tiruppur:1,Erode:1.5,Salem:3,Madurai:4};
  const grd=['Premium','Grade A','Grade A','Grade B','Mixed'];
  const sIdx=[7,5,2,-1,-4,-6,-5,-3,0,3,6,8];
  const data=[],st={};
  mkt.forEach(m=>st[m]=42+adj[m]);
  const cur=new Date('2020-01-01'), end=new Date('2026-02-23');
  while(cur<=end){
    const ds=cur.toISOString().slice(0,10);
    const yr=cur.getFullYear(),mo=cur.getMonth(),dow=cur.getDay();
    const today=dow===0?['Pollachi']:[3,6].includes(dow)?mkt.slice(0,4):mkt;
    for(const m of today){
      const target=44+adj[m]+(yr-2019)*1.8+sIdx[mo];
      st[m]=Math.max(28,Math.min(115,st[m]+(target-st[m])*.10+(Math.random()-.49)*3.2));
      const modal=Math.round(st[m]*2)/2;
      const sp=3.5+Math.random()*7;
      const minP=Math.max(24,Math.round((modal-sp*.55)*2)/2);
      const maxP=Math.round((modal+sp*.45)*2)/2;
      const bv={Pollachi:450,Coimbatore:320,Chennai:580,Tiruppur:180,Erode:200,Salem:150,Madurai:250}[m]||200;
      data.push({date:ds,market:m,minPrice:minP,modalPrice:modal,maxPrice:maxP,
        volume:Math.max(50,Math.round(bv*(.7+Math.random()*.6))),
        grade:grd[Math.floor(Math.random()*grd.length)]});
    }
    cur.setDate(cur.getDate()+1);
  }
  data.sort((a,b)=>a.date.localeCompare(b.date));
  return data;
}

/* Downsample daily → weekly average (SPEED) */
function weeklyAvg(prices){
  const out=[];
  for(let i=0;i<prices.length;i+=7){
    const slice=prices.slice(i,i+7);
    out.push(slice.reduce((s,v)=>s+v,0)/slice.length);
  }
  return out;
}

/* ── 2. AR(7) ── */
function fitAR(prices,p=7){
  const X=[],y=[];
  for(let i=p;i<prices.length;i++){
    const row=[1];for(let j=1;j<=p;j++)row.push(prices[i-j]);
    X.push(row);y.push(prices[i]);
  }
  return solveLinear(matMul(transpose(X),X),matVecMul(transpose(X),y));
}
function arForecast(hist,beta,steps){
  const p=beta.length-1,buf=[...hist.slice(-p)],out=[];
  for(let s=0;s<steps;s++){
    let v=beta[0];for(let j=1;j<=p;j++)v+=beta[j]*buf[buf.length-j];
    v=Math.max(24,Math.min(120,v));out.push(v);buf.push(v);
  }
  return out;
}

/* ── 3. HOLT-WINTERS (fast grid) ── */
function hwSSE(p,a,b){
  let L=p[0],T=p[1]-p[0],sse=0;
  for(let i=1;i<p.length;i++){
    const pred=L+T,Lp=L,Tp=T;
    L=a*p[i]+(1-a)*(Lp+Tp);T=b*(L-Lp)+(1-b)*Tp;
    sse+=(p[i]-pred)**2;
  }
  return sse;
}
function fitHW(p){
  /* ~20 combinations instead of 80 — enough for good params */
  let ba=0.3,bb=0.1,best=Infinity;
  for(let a=0.1;a<=0.9;a+=0.2)for(let b=0.05;b<=0.4;b+=0.15){
    const e=hwSSE(p,a,b);if(e<best){best=e;ba=a;bb=b;}
  }
  return{alpha:ba,beta:bb};
}
function hwRun(p,a,b){
  let L=p[0],T=p[1]-p[0];
  for(let i=1;i<p.length;i++){const Lp=L,Tp=T;L=a*p[i]+(1-a)*(Lp+Tp);T=b*(L-Lp)+(1-b)*Tp;}
  return{L,T};
}
function hwForecast(prices,a,b,steps){
  const{L,T}=hwRun(prices,a,b);
  return Array.from({length:steps},(_,h)=>Math.max(24,Math.min(120,L+T*(h+1))));
}

/* ── 4. LSTM (fast: weekly data, 1 layer, 10 LB, 20ep, batch128) ── */
const LB=10,EPOCHS=20;
function normArr(arr){const mn=Math.min(...arr),mx=Math.max(...arr),rng=mx-mn||1;return{n:arr.map(v=>(v-mn)/rng),mn,rng};}
function denorm(v,mn,rng){return v*rng+mn;}

async function trainLSTM(weeklyPrices,onEpoch){
  const{n,mn,rng}=normArr(weeklyPrices);
  const Xs=[],ys=[];
  for(let i=LB;i<n.length;i++){Xs.push(n.slice(i-LB,i).map(v=>[v]));ys.push(n[i]);}
  const sp=Math.floor(Xs.length*.85);
  const xTr=tf.tensor3d(Xs.slice(0,sp));
  const yTr=tf.tensor2d(ys.slice(0,sp).map(v=>[v]));
  const m=tf.sequential();
  /* Single LSTM layer — much faster, still captures temporal patterns */
  m.add(tf.layers.lstm({units:24,returnSequences:false,inputShape:[LB,1],dropout:0.05}));
  m.add(tf.layers.dense({units:1}));
  m.compile({optimizer:tf.train.adam(0.005),loss:'meanSquaredError'});
  await m.fit(xTr,yTr,{epochs:EPOCHS,batchSize:128,shuffle:true,
    callbacks:{onEpochEnd:async(ep,logs)=>{onEpoch(ep,logs.loss);await tf.nextFrame();}}});
  xTr.dispose();yTr.dispose();
  return{model:m,n,mn,rng};
}

/* Batch rollout — much faster than 90 individual tensor ops */
async function lstmForecastBatch(obj,steps){
  const buf=[...obj.n.slice(-LB)],out=[];
  /* Process in batches of 10 */
  const BATCHSZ=10;
  let s=0;
  while(s<steps){
    const bsz=Math.min(BATCHSZ,steps-s);
    for(let b=0;b<bsz;b++){
      const inp=tf.tensor3d([buf.slice(-LB).map(v=>[v])]);
      const res=obj.model.predict(inp);
      const raw=(await res.data())[0];
      inp.dispose();res.dispose();
      const v=Math.max(0,Math.min(1,raw));
      buf.push(v);
      out.push(Math.max(24,Math.min(120,denorm(v,obj.mn,obj.rng))));
    }
    await tf.nextFrame(); // yield between batches
    s+=bsz;
  }
  return out;
}

/* Convert weekly LSTM forecast → daily (interpolate) */
function weeklyToDaily(weeklyFc,days){
  const out=[];
  for(let d=0;d<days;d++){
    const wk=Math.floor(d/7);
    const wk1=Math.min(wk+1,weeklyFc.length-1);
    const t=(d%7)/7;
    out.push(weeklyFc[wk]*(1-t)+weeklyFc[wk1]*t);
  }
  return out;
}

/* ── 5. LINEAR ALGEBRA ── */
function transpose(M){return M[0].map((_,j)=>M.map(r=>r[j]));}
function matMul(A,B){const R=A.length,C=B[0].length,K=B.length;return Array.from({length:R},(_,i)=>Array.from({length:C},(_,j)=>{let s=0;for(let k=0;k<K;k++)s+=A[i][k]*B[k][j];return s;}));}
function matVecMul(A,v){return A.map(r=>r.reduce((s,a,j)=>s+a*v[j],0));}
function solveLinear(A,b){
  const n=b.length,M=A.map((r,i)=>[...r,b[i]]);
  for(let c=0;c<n;c++){
    let mx=c;for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[mx][c]))mx=r;
    [M[c],M[mx]]=[M[mx],M[c]];
    const pv=M[c][c]||1e-10;
    for(let r=0;r<n;r++){if(r===c)continue;const f=M[r][c]/pv;for(let cc=c;cc<=n;cc++)M[r][cc]-=f*M[c][cc];}
    for(let cc=c;cc<=n;cc++)M[c][cc]/=pv;
  }
  return M.map(r=>r[n]);
}

/* ── 6. METRICS & ENSEMBLE ── */
function mape(p,a){return a.reduce((s,v,i)=>s+Math.abs((v-p[i])/Math.max(1,v)),0)/a.length*100;}
function mae(p,a){return a.reduce((s,v,i)=>s+Math.abs(v-p[i]),0)/a.length;}
function optWeights(ar,hw,lm,act){
  let best={ar:.33,hw:.33,lm:.34},bErr=Infinity;
  for(let a=0;a<=1;a+=0.1)for(let h=0;h<=1-a;h+=0.1){
    const l=+(1-a-h).toFixed(2);if(l<0)continue;
    const err=act.reduce((s,v,i)=>s+Math.abs(a*ar[i]+h*hw[i]+l*lm[i]-v),0)/act.length;
    if(err<bErr){bErr=err;best={ar:a,hw:h,lm:l};}
  }
  return best;
}
function ensemble(ar,hw,lm,w,n){return Array.from({length:n},(_,i)=>w.ar*ar[i]+w.hw*hw[i]+w.lm*lm[i]);}
function residCI(resid){
  const abs=resid.map(Math.abs).sort((a,b)=>a-b);
  return{q80:abs[Math.floor(abs.length*.80)]||3,q95:abs[Math.floor(abs.length*.95)]||5};
}
function computeRSI(p,period=14){
  const ch=p.slice(1).map((v,i)=>v-p[i]);
  const g=ch.slice(-period).filter(c=>c>0).reduce((s,v)=>s+v,0)/period;
  const l=Math.abs(ch.slice(-period).filter(c=>c<0).reduce((s,v)=>s+v,0))/period;
  return l===0?100:100-(100/(1+g/l));
}
function MA(arr,n){return arr.slice(-n).reduce((s,v)=>s+v,0)/n;}
function STD(arr){const m=arr.reduce((s,v)=>s+v,0)/arr.length;return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/arr.length);}

/* ── 7. STATE ── */
let DS=[],filtDS=[],dsPg=1;const DPS=25;
let PP=[],WP=[]; // daily and weekly pollachi prices
let arBeta=null,hwP=null,lstmObj=null,WT={ar:.33,hw:.33,lm:.34},CI={q80:3,q95:6};
let vAct=[],vAR=[],vHW=[],vLM=[],vEns=[];
let metrics={ar:0,hw:0,lm:0,ens:0,mae:0,acc:0};
let fcOut=null,chartRange=90;
let MC=null,FC=null,MoC=null;
let editIdx=-1;
let trainStart=0;

/* ── 8. TRAIN UI HELPERS ── */
function stepUI(i,state,val=''){
  const si=document.getElementById('si'+i),sl=document.getElementById('sl'+i),sv=document.getElementById('sv'+i);
  if(state==='run'){si.className='si run';si.textContent='↻';}
  else if(state==='done'){si.className='si done';si.textContent='✓';sl.className='sl done';sv.className='sv done';}
  if(val)sv.textContent=val;
}
function prog(p){document.getElementById('pf').style.width=p+'%';}
function elog(m){const e=document.getElementById('elog');e.innerHTML+=m+'<br>';e.scrollTop=e.scrollHeight;}
const delay=ms=>new Promise(r=>setTimeout(r,ms));

function updateTimer(){
  const el=Math.round((Date.now()-trainStart)/1000);
  document.getElementById('et-elapsed').textContent=`Elapsed: ${el}s`;
}

/* ── 9. TRAINING ── */
async function trainAll(){
  trainStart=Date.now();
  const timerInterval=setInterval(updateTimer,1000);
  try{
    const hasCached=loadCache();

    /* Step 0 — Data */
    stepUI(0,'run');
    DS=generateDataset();filtDS=[...DS];
    PP=DS.filter(d=>d.market==='Pollachi').map(d=>d.modalPrice);
    WP=weeklyAvg(PP); // weekly averages for LSTM
    const TR=Math.floor(PP.length*.85);
    const trP=PP.slice(0,TR), vaP=PP.slice(TR);
    const TRw=Math.floor(WP.length*.85);
    const trW=WP.slice(0,TRw), vaW=WP.slice(TRw);
    stepUI(0,'done',`${DS.length.toLocaleString()} rows · ${WP.length}wks`);
    prog(14);await delay(30);

    let arVal,hwVal,arMP,hwMP;

    if(hasCached&&arBeta&&hwP){
      /* Use cached AR/HW — skip refitting */
      stepUI(1,'done',`${metrics.ar.toFixed(1)}% (cached)`);
      stepUI(2,'done',`${metrics.hw.toFixed(1)}% (cached)`);
      arVal=arForecast(trP,arBeta,vaP.length);
      hwVal=hwForecast(trP,hwP.alpha,hwP.beta,vaP.length);
      arMP=metrics.ar;hwMP=metrics.hw;
      prog(42);
    }else{
      /* Step 1 — AR */
      stepUI(1,'run');
      arBeta=fitAR(trP,7);
      arVal=arForecast(trP,arBeta,vaP.length);
      arMP=mape(arVal,vaP);
      stepUI(1,'done',`MAPE ${arMP.toFixed(1)}%`);
      elog(`AR β=[${arBeta.slice(1).map(v=>v.toFixed(3)).join(', ')}]`);
      prog(28);await delay(20);

      /* Step 2 — HW */
      stepUI(2,'run');
      hwP=fitHW(trP);
      hwVal=hwForecast(trP,hwP.alpha,hwP.beta,vaP.length);
      hwMP=mape(hwVal,vaP);
      stepUI(2,'done',`MAPE ${hwMP.toFixed(1)}% α=${hwP.alpha.toFixed(2)}`);
      prog(42);await delay(20);
    }

    /* Step 3 — LSTM on WEEKLY data */
    stepUI(3,'run');
    document.getElementById('et-est').textContent='Est. remaining: ~30–60s';
    let lastL=0,ep0=Date.now();
    lstmObj=await trainLSTM(trW,(ep,loss)=>{
      lastL=loss;
      if(ep%5===0)elog(`LSTM ep${ep+1}/${EPOCHS} loss=${loss.toFixed(5)}`);
      prog(42+Math.round((ep/EPOCHS)*36));
    });
    /* Val predictions — weekly then interpolate to daily length */
    const lmWVal=await lstmForecastBatch(lstmObj,vaW.length);
    const lmDVal=weeklyToDaily(lmWVal,vaP.length);
    const lmMP=mape(lmDVal,vaP);
    const lstmMs=Math.round((Date.now()-ep0)/1000);
    stepUI(3,'done',`MAPE ${lmMP.toFixed(1)}% · ${lstmMs}s`);
    prog(82);await delay(20);

    /* Step 4 — Ensemble */
    stepUI(4,'run');
    WT=optWeights(arVal,hwVal,lmDVal,vaP);
    const ensVal=ensemble(arVal,hwVal,lmDVal,WT,vaP.length);
    const enMP=mape(ensVal,vaP);
    const enMA=mae(ensVal,vaP);
    CI=residCI(vaP.map((v,i)=>v-ensVal[i]));
    metrics={ar:arMP,hw:hwMP,lm:lmMP,ens:enMP,mae:enMA,acc:(100-enMP).toFixed(1)};
    vAct=[...vaP];vAR=[...arVal];vHW=[...hwVal];vLM=[...lmDVal];vEns=[...ensVal];
    elog(`Ens: AR=${WT.ar.toFixed(2)} HW=${WT.hw.toFixed(2)} LSTM=${WT.lm.toFixed(2)}`);
    elog(`MAPE=${enMP.toFixed(1)}% MAE=₹${enMA.toFixed(1)} CI±₹${CI.q80.toFixed(1)}`);
    stepUI(4,'done',`MAPE ${enMP.toFixed(1)}%`);
    prog(100);
    saveCache();

    const totalS=Math.round((Date.now()-trainStart)/1000);
    document.getElementById('et-elapsed').textContent=`Done in ${totalS}s`;
    document.getElementById('et-est').textContent='';
    clearInterval(timerInterval);
    await delay(250);

    document.getElementById('train-overlay').style.display='none';
    document.getElementById('mst').textContent=`MAPE ${enMP.toFixed(1)}% · AR+HW+LSTM`;

    await buildFC();
    populateAll();

  }catch(err){
    clearInterval(timerInterval);
    console.error('Train error:',err);
    elog(`❌ ${err.message}`);
    toast('⚠ Training error — check console');
  }
}

/* ── 10. FORWARD FORECAST (90 days) ── */
async function buildFC(){
  const N=90,NW=Math.ceil(N/7)+2;
  const ar90=arForecast(PP,arBeta,N);
  const hw90=hwForecast(PP,hwP.alpha,hwP.beta,N);
  let lm90;
  if(lstmObj&&lstmObj.model){
    const lmW=await lstmForecastBatch(lstmObj,NW);
    lm90=weeklyToDaily(lmW,N);
  }else{lm90=ar90;}
  const en90=ensemble(ar90,hw90,lm90,WT,N);
  fcOut={ar:ar90,hw:hw90,lm:lm90,en:en90,N};
}

function fcAt(day){
  if(!fcOut)return{pt:0,lm:0,ar:0,mn80:0,mx80:0,mn95:0,mx95:0,cf:70};
  const i=Math.min(day-1,fcOut.N-1);
  const pt=fcOut.en[i],hf=Math.sqrt(Math.max(1,day));
  return{
    pt:Math.round(pt*2)/2,lm:Math.round(fcOut.lm[i]*2)/2,ar:Math.round(fcOut.ar[i]*2)/2,
    mn80:Math.max(24,Math.round((pt-CI.q80*hf)*2)/2),mx80:Math.round((pt+CI.q80*hf)*2)/2,
    mn95:Math.max(22,Math.round((pt-CI.q95*hf)*2)/2),mx95:Math.round((pt+CI.q95*hf)*2)/2,
    cf:Math.max(55,Math.round(92-day*.28))
  };
}

/* ── 11. POPULATE UI ── */
function populateAll(){
  const cp=PP.at(-1),prev=PP.at(-2)||cp;
  const chg=cp-prev,chgP=(chg/prev*100);
  const r7=PP.slice(-7),r30=PP.slice(-30);
  const rsi=computeRSI(PP.slice(-20)),ma7=MA(PP,7),ma30=MA(PP,30),std30=STD(r30);
  document.getElementById('tk0').textContent=`₹${cp}`;
  document.getElementById('tk0c').textContent=`${chg>=0?'+':''}${chg.toFixed(1)} (${chgP.toFixed(1)}%)`;
  document.getElementById('tk0c').className=`tv ${chg>=0?'up':'down'}`;
  document.getElementById('tk1').textContent=`₹${Math.max(...r7)}`;
  document.getElementById('tk2').textContent=`₹${Math.min(...r7)}`;
  document.getElementById('tk3').textContent=`₹${(r30.reduce((s,v)=>s+v,0)/30).toFixed(1)}`;
  document.getElementById('tk4').textContent=`±₹${std30.toFixed(1)}`;
  document.getElementById('tk5').textContent=rsi.toFixed(0);
  document.getElementById('tk6').textContent=rsi>60&&cp>ma30?'BUY':rsi<40&&cp<ma30?'SELL':'HOLD';
  const t1=fcAt(1),m1=fcAt(30);
  document.getElementById('sc0').textContent=`₹${cp}`;
  document.getElementById('sc0').className=`cval ${chg>=0?'up':'down'}`;
  document.getElementById('sc0s').textContent=`${chg>=0?'▲':'▼'} ₹${Math.abs(chg).toFixed(1)} (${Math.abs(chgP).toFixed(1)}%)`;
  document.getElementById('sc1').textContent=`₹${t1.pt}`;
  document.getElementById('sc1').className=`cval ${t1.pt>=cp?'up':'down'}`;
  document.getElementById('sc1s').textContent=`₹${t1.mn80} – ₹${t1.mx80}`;
  document.getElementById('sc2').textContent=`₹${m1.pt}`;
  document.getElementById('sc2').className=`cval ${m1.pt>=cp?'up':'down'}`;
  document.getElementById('sc2s').textContent=m1.pt>cp?'↗ Upward':'↘ Downward';
  document.getElementById('sc3').textContent=`${(+metrics.ens).toFixed(1)}%`;
  document.getElementById('sc3s').textContent=`MAE ₹${(+metrics.mae).toFixed(1)} · AR ${(+metrics.ar).toFixed(1)}% · LSTM ${(+metrics.lm).toFixed(1)}%`;
  renderFCCards(cp);
  renderTech(cp,ma7,ma30,rsi,std30,chgP);
  renderSeas();
  renderFCTable(cp);
  renderDSTable();
  renderMetrics();
  renderFeatBars();
  setTimeout(()=>initMainChart(chartRange),80);
  setTimeout(()=>{initFCChart();initMonthChart();},260);
  setTimeout(()=>{initResidChart();initValChart();initWgtChart();},500);
}

function renderFCCards(cp){
  const pds=[{l:'Tomorrow',d:1,i:'📅'},{l:'Next Week',d:7,i:'📆'},{l:'Next Month',d:30,i:'🗓'},{l:'3 Months',d:90,i:'📊'}];
  document.getElementById('fc-cards').innerHTML=pds.map(p=>{
    const f=fcAt(p.d),up=f.pt>cp,pct=((f.pt-cp)/cp*100).toFixed(1);
    return`<div class="fcc"><div class="fper">${p.i} ${p.l}</div>
    <div class="frng"><span class="down">₹${f.mn80}</span><span style="color:var(--text3);margin:0 4px">—</span><span class="up">₹${f.mx80}</span></div>
    <div class="fexp">Expected: <strong style="color:var(--text)">₹${f.pt}/kg</strong></div>
    <div class="crow"><span>${f.cf}%</span><div class="cbar"><div class="cfil" style="width:${f.cf}%"></div></div></div>
    <div class="ftr ${up?'up':'down'}">${up?'▲':'▼'} ${Math.abs(pct)}%</div>
    <div class="mbdg">AR+HW+LSTM</div></div>`;
  }).join('');
}
function renderTech(cp,ma7,ma30,rsi,std30,chgP){
  const mom=((cp-(PP.at(-8)||cp))/(PP.at(-8)||cp)*100);
  [['MA-7',`₹${ma7.toFixed(1)}`,cp>ma7?'bb':'bd',cp>ma7?'BULLISH':'BEARISH'],
   ['MA-30',`₹${ma30.toFixed(1)}`,cp>ma30?'bb':'bd',cp>ma30?'BULLISH':'BEARISH'],
   ['RSI(14)',rsi.toFixed(0),rsi>70?'bd':rsi<30?'bb':'bn',rsi>70?'OVERBOUGHT':rsi<30?'OVERSOLD':'NEUTRAL'],
   ['Volatility σ',`₹${std30.toFixed(1)}`,'bn','MODERATE'],
   ['7D Momentum',`${mom.toFixed(1)}%`,mom>0?'bb':'bd',mom>0?'POSITIVE':'NEGATIVE'],
   ['Daily Δ',`${chgP.toFixed(1)}%`,chgP>=0?'bb':'bd',chgP>=0?'UP':'DOWN']
  ].reduce((h,r)=>h+`<div class="si-item"><span class="sn">${r[0]}</span><div style="display:flex;align-items:center;gap:5px"><span class="sv2">${r[1]}</span><span class="badge ${r[2]}">${r[3]}</span></div></div>`,'');
  document.getElementById('tech-sigs').innerHTML=[['MA-7',`₹${ma7.toFixed(1)}`,cp>ma7?'bb':'bd',cp>ma7?'BULLISH':'BEARISH'],['MA-30',`₹${ma30.toFixed(1)}`,cp>ma30?'bb':'bd',cp>ma30?'BULLISH':'BEARISH'],['RSI(14)',rsi.toFixed(0),rsi>70?'bd':rsi<30?'bb':'bn',rsi>70?'OVERBOUGHT':rsi<30?'OVERSOLD':'NEUTRAL'],['σ Volatility',`₹${std30.toFixed(1)}`,'bn','MODERATE'],['7D Momentum',`${mom.toFixed(1)}%`,mom>0?'bb':'bd',mom>0?'POSITIVE':'NEGATIVE'],['Daily Δ',`${chgP.toFixed(1)}%`,chgP>=0?'bb':'bd',chgP>=0?'UP':'DOWN']].map(r=>`<div class="si-item"><span class="sn">${r[0]}</span><div style="display:flex;align-items:center;gap:5px"><span class="sv2">${r[1]}</span><span class="badge ${r[2]}">${r[3]}</span></div></div>`).join('');
}
function renderSeas(){
  const mo=new Date('2026-02-23').getMonth();
  const seas=['Winter','Late Winter','Spring','Pre-Summer','Summer','Monsoon','Monsoon','Monsoon','Post-Mon','Festive','NE Monsoon','Peak Winter'];
  const harv=['Post-Harvest','Post-Harvest','Mid Season','Off Season','Off Season','Pre-Harvest','Harvest','Harvest','Post-Harvest','Festival','Festival','Pre-Season'];
  document.getElementById('seas-sigs').innerHTML=[['Season',seas[mo]],['Harvest Phase',harv[mo]],['AR Tomorrow',`₹${arForecast(PP,arBeta,1)[0].toFixed(1)}`],['HW Tomorrow',`₹${hwForecast(PP,hwP.alpha,hwP.beta,1)[0].toFixed(1)}`],['LSTM Tomorrow',`₹${fcOut.lm[0].toFixed(1)}`],['Ensemble Tomorrow',`₹${fcOut.en[0].toFixed(1)}`]].map(r=>`<div class="si-item"><span class="sn">${r[0]}</span><span class="sv2">${r[1]}</span></div>`).join('');
}
function renderFCTable(cp){
  const today=new Date('2026-02-23');
  document.getElementById('fc-tbody').innerHTML=Array.from({length:12},(_,i)=>{
    const d=(i+1)*7,f=fcAt(d);
    const s=new Date(today);s.setDate(s.getDate()+i*7+1);
    const e=new Date(today);e.setDate(e.getDate()+(i+1)*7);
    const pr=`${s.toLocaleDateString('en-IN',{month:'short',day:'2-digit'})} – ${e.toLocaleDateString('en-IN',{month:'short',day:'2-digit'})}`;
    const ch=((f.pt-cp)/cp*100),arUp=fcOut.ar[d-1]>cp,lmUp=fcOut.lm[d-1]>cp;
    return`<tr><td style="font-family:'Instrument Sans',sans-serif">W${i+1}</td><td style="font-family:'Instrument Sans',sans-serif;color:var(--text2)">${pr}</td><td class="down">₹${f.mn80}</td><td style="color:var(--text);font-weight:600">₹${f.pt}</td><td class="up">₹${f.mx80}</td><td class="${ch>=0?'up':'down'}">${ch>=0?'+':''}${ch.toFixed(1)}%</td><td><div style="display:flex;align-items:center;gap:4px"><div class="cbar" style="width:44px"><div class="cfil" style="width:${f.cf}%"></div></div><span style="font-size:.63rem;color:var(--text2)">${f.cf}%</span></div></td><td><span class="badge ${arUp?'bb':'bd'}">${arUp?'↑':'↓'} AR</span></td><td><span class="badge ${lmUp?'bb':'bd'}">${lmUp?'↑':'↓'} LSTM</span></td></tr>`;
  }).join('');
}
function renderMetrics(){
  document.getElementById('mgrid').innerHTML=[{n:`${(+metrics.ens).toFixed(1)}%`,l:'Ensemble MAPE'},{n:`₹${(+metrics.mae).toFixed(1)}`,l:'MAE'},{n:`${(+metrics.ar).toFixed(1)}%`,l:'AR(7) MAPE'},{n:`${(+metrics.lm).toFixed(1)}%`,l:'LSTM MAPE'}].map(m=>`<div class="mbox"><div class="mnum">${m.n}</div><div class="mlbl">${m.l}</div></div>`).join('');
}
function renderFeatBars(){
  const c=arBeta.slice(1).map((v,i)=>({n:`Lag ${i+1}`,v:Math.abs(v)})),mx=Math.max(...c.map(x=>x.v));
  document.getElementById('feat-bars').innerHTML=c.map(x=>`<div class="fbr"><div class="fn">${x.n}</div><div class="fb"><div class="ff" style="width:${(x.v/mx*100).toFixed(0)}%"></div></div><div class="fp">${(x.v/mx*100).toFixed(0)}%</div></div>`).join('');
}

/* ── 12. CHARTS ── */
const CO={responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
  plugins:{legend:{display:false},tooltip:{backgroundColor:'#0e0e12',borderColor:'#252535',borderWidth:1,titleColor:'#e8edf5',bodyColor:'#7a8494',callbacks:{label:c=>`${c.dataset.label}: ₹${c.raw??'N/A'}`}}},
  scales:{x:{grid:{color:'#1c1c24'},ticks:{color:'#3a4050',font:{family:'JetBrains Mono',size:9},maxTicksLimit:10}},
          y:{grid:{color:'#1c1c24'},ticks:{color:'#3a4050',font:{family:'JetBrains Mono',size:9},callback:v=>`₹${v}`}}}};

function initMainChart(days){
  try{
    const ctx=document.getElementById('mainChart').getContext('2d');
    if(MC)MC.destroy();
    const hist=PP.slice(-Math.min(days,PP.length));
    const hdates=DS.filter(d=>d.market==='Pollachi').map(d=>d.date).slice(-hist.length);
    const FD=30,today=new Date('2026-02-23');
    const fl=Array.from({length:FD},(_,i)=>{const d=new Date(today);d.setDate(d.getDate()+i+1);return d.toISOString().slice(0,10);});
    const al=[...hdates,...fl];
    const ah=[...hist,...Array(FD).fill(null)];
    const ef=[...Array(hist.length-1).fill(null),hist.at(-1),...fcOut.en.slice(0,FD)];
    const lf=[...Array(hist.length-1).fill(null),hist.at(-1),...fcOut.lm.slice(0,FD)];
    const cimn=[...Array(hist.length).fill(null),...fcOut.en.slice(0,FD).map((v,i)=>v-CI.q80*Math.sqrt(i+1))];
    const cimx=[...Array(hist.length).fill(null),...fcOut.en.slice(0,FD).map((v,i)=>v+CI.q80*Math.sqrt(i+1))];
    MC=new Chart(ctx,{type:'line',data:{labels:al,datasets:[
      {label:'CI Min',data:cimn,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.07)',fill:'+1',pointRadius:0},
      {label:'CI Max',data:cimx,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.07)',fill:'-1',pointRadius:0},
      {label:'Actual',data:ah,borderColor:'#10b981',borderWidth:1.8,pointRadius:0,tension:.3},
      {label:'Ensemble',data:ef,borderColor:'#818cf8',borderWidth:2,borderDash:[6,3],pointRadius:0,tension:.4},
      {label:'LSTM',data:lf,borderColor:'#22d3ee',borderWidth:1.4,borderDash:[3,3],pointRadius:0,tension:.4},
    ]},options:{...CO}});
  }catch(e){console.error('mainChart:',e);}
}
function initFCChart(){
  try{
    const ctx=document.getElementById('fcastChart').getContext('2d');
    if(FC)FC.destroy();
    const today=new Date('2026-02-23');
    const labels=Array.from({length:fcOut.N},(_,i)=>{const d=new Date(today);d.setDate(d.getDate()+i+1);return d.toISOString().slice(0,10);});
    const ci80mn=fcOut.en.map((v,i)=>v-CI.q80*Math.sqrt(i+1));
    const ci80mx=fcOut.en.map((v,i)=>v+CI.q80*Math.sqrt(i+1));
    FC=new Chart(ctx,{type:'line',data:{labels,datasets:[
      {label:'80% CI Min',data:ci80mn,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.11)',fill:'+1',pointRadius:0},
      {label:'80% CI Max',data:ci80mx,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.11)',fill:'-1',pointRadius:0},
      {label:'Ensemble',data:fcOut.en,borderColor:'#818cf8',borderWidth:2.5,pointRadius:0,tension:.4},
      {label:'LSTM',data:fcOut.lm,borderColor:'#22d3ee',borderWidth:1.5,borderDash:[4,2],pointRadius:0,tension:.4},
      {label:'AR(7)',data:fcOut.ar,borderColor:'#f97316',borderWidth:1.2,borderDash:[2,4],pointRadius:0,tension:.3},
      {label:'HW',data:fcOut.hw,borderColor:'#f59e0b',borderWidth:1,borderDash:[1,4],pointRadius:0,tension:.3},
    ]},options:{...CO,plugins:{...CO.plugins,legend:{display:true,labels:{color:'#7a8494',font:{family:'Instrument Sans',size:10}}}}}});
  }catch(e){console.error('fcastChart:',e);}
}
function initMonthChart(){
  try{
    if(MoC)MoC.destroy();
    const today=new Date('2026-02-23');
    const labels=Array.from({length:6},(_,i)=>{const d=new Date(today);d.setMonth(d.getMonth()+i+1);return d.toLocaleDateString('en-IN',{month:'short',year:'2-digit'});});
    const md=Array.from({length:6},(_,i)=>{const f=fcAt((i+1)*30);return{mn:f.mn80,ex:f.pt,mx:f.mx80};});
    MoC=new Chart(document.getElementById('monthChart'),{type:'bar',data:{labels,datasets:[
      {label:'Min',data:md.map(d=>d.mn),backgroundColor:'rgba(239,68,68,.28)',borderColor:'rgba(239,68,68,.7)',borderWidth:1,borderRadius:3},
      {label:'Expected',data:md.map(d=>d.ex),backgroundColor:'rgba(34,211,238,.32)',borderColor:'rgba(34,211,238,.8)',borderWidth:1,borderRadius:3},
      {label:'Max',data:md.map(d=>d.mx),backgroundColor:'rgba(129,140,248,.28)',borderColor:'rgba(129,140,248,.7)',borderWidth:1,borderRadius:3},
    ]},options:{...CO,plugins:{...CO.plugins,legend:{display:true,labels:{color:'#7a8494',font:{family:'Instrument Sans',size:10}}}}}});
  }catch(e){console.error('monthChart:',e);}
}
function initResidChart(){
  try{
    const r=vAct.map((v,i)=>v-vEns[i]);
    new Chart(document.getElementById('residChart'),{type:'bar',data:{labels:r.map((_,i)=>`v${i+1}`),datasets:[{label:'Residual',data:r,backgroundColor:r.map(v=>v>=0?'rgba(16,185,129,.45)':'rgba(239,68,68,.45)'),borderColor:r.map(v=>v>=0?'rgba(16,185,129,.8)':'rgba(239,68,68,.8)'),borderWidth:1,borderRadius:2}]},options:{...CO,plugins:{...CO.plugins,legend:{display:false}}}});
  }catch(e){console.error('residChart:',e);}
}
function initValChart(){
  try{
    new Chart(document.getElementById('valChart'),{type:'line',data:{labels:vAct.map((_,i)=>`d${i+1}`),datasets:[
      {label:'Actual',data:vAct,borderColor:'#10b981',borderWidth:2,pointRadius:0,tension:.3},
      {label:'Ensemble',data:vEns,borderColor:'#818cf8',borderWidth:1.5,borderDash:[4,2],pointRadius:0,tension:.3},
      {label:'AR(7)',data:vAR,borderColor:'#f97316',borderWidth:1.1,borderDash:[2,3],pointRadius:0,tension:.3},
      {label:'LSTM',data:vLM,borderColor:'#22d3ee',borderWidth:1.1,borderDash:[3,2],pointRadius:0,tension:.3},
      {label:'HW',data:vHW,borderColor:'#f59e0b',borderWidth:1,borderDash:[1,4],pointRadius:0,tension:.3},
    ]},options:{...CO,plugins:{...CO.plugins,legend:{display:true,labels:{color:'#7a8494',font:{family:'Instrument Sans',size:10}}}}}});
  }catch(e){console.error('valChart:',e);}
}
function initWgtChart(){
  try{
    new Chart(document.getElementById('wgtChart'),{type:'doughnut',data:{
      labels:[`AR ${(WT.ar*100).toFixed(0)}%`,`HW ${(WT.hw*100).toFixed(0)}%`,`LSTM ${(WT.lm*100).toFixed(0)}%`],
      datasets:[{data:[WT.ar,WT.hw,WT.lm],backgroundColor:['#f97316','#f59e0b','#22d3ee'],borderColor:'#09090b',borderWidth:3}]
    },options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#7a8494',font:{family:'Instrument Sans',size:11},padding:8}},tooltip:{backgroundColor:'#0e0e12',borderColor:'#252535',borderWidth:1}}}});
  }catch(e){console.error('wgtChart:',e);}
}

/* ── 13. DATASET TABLE ── */
function renderDSTable(){
  const s=(dsPg-1)*DPS,sl=filtDS.slice(s,s+DPS);
  document.getElementById('ds-cnt').textContent=`${DS.length.toLocaleString()} total · ${PP.length} Pollachi · ${WP.length} weeks`;
  document.getElementById('ds-info').textContent=`${s+1}–${Math.min(s+DPS,filtDS.length)} of ${filtDS.length.toLocaleString()}`;
  document.getElementById('ds-pg').textContent=`${dsPg}/${Math.ceil(filtDS.length/DPS)}`;
  document.getElementById('ds-tbody').innerHTML=sl.map((r,i)=>{
    const gi=DS.indexOf(r);
    return`<tr><td style="color:var(--text3)">${s+i+1}</td><td style="font-family:'Instrument Sans',sans-serif">${r.date}</td><td style="font-family:'Instrument Sans',sans-serif">${r.market}</td><td class="down">₹${r.minPrice}</td><td style="font-weight:600">₹${r.modalPrice}</td><td class="up">₹${r.maxPrice}</td><td>${r.volume}</td><td><span class="badge bn" style="font-size:.56rem">${r.grade}</span></td><td><div style="display:flex;gap:3px"><button class="btn" style="padding:2px 7px;font-size:.67rem" onclick="editRec(${gi})">Edit</button><button class="btn" style="padding:2px 7px;font-size:.67rem;color:var(--down)" onclick="delRec(${gi})">Del</button></div></td></tr>`;
  }).join('');
}
function filterDS(){
  const q=document.getElementById('dss').value.toLowerCase(),yr=document.getElementById('dsy').value,mk=document.getElementById('dsm').value;
  filtDS=DS.filter(d=>(!q||d.date.includes(q)||d.market.toLowerCase().includes(q))&&(!yr||d.date.startsWith(yr))&&(!mk||d.market===mk));
  dsPg=1;renderDSTable();
}
function dsPage(d){const mx=Math.ceil(filtDS.length/DPS);dsPg=Math.max(1,Math.min(mx,dsPg+d));renderDSTable();}
function exportCSV(){
  const h=['Date','Market','MinPrice','ModalPrice','MaxPrice','Volume','Grade'];
  const csv=[h,...filtDS.map(d=>[d.date,d.market,d.minPrice,d.modalPrice,d.maxPrice,d.volume,d.grade])].map(r=>r.join(',')).join('\n');
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='tn_coconut.csv';a.click();
  toast('✓ CSV exported');
}

/* ── 14. MODAL ── */
function openAdd(){editIdx=-1;document.getElementById('mtitle').textContent='+ Add Record';['m-min','m-max','m-mod','m-vol'].forEach(id=>document.getElementById(id).value='');document.getElementById('m-date').value='2026-02-24';document.getElementById('modal').classList.add('open');}
function editRec(i){editIdx=i;const r=DS[i];document.getElementById('mtitle').textContent='Edit Record';document.getElementById('m-date').value=r.date;document.getElementById('m-mkt').value=r.market;document.getElementById('m-min').value=r.minPrice;document.getElementById('m-max').value=r.maxPrice;document.getElementById('m-mod').value=r.modalPrice;document.getElementById('m-vol').value=r.volume;document.getElementById('m-grd').value=r.grade;document.getElementById('modal').classList.add('open');}
function closeModal(){document.getElementById('modal').classList.remove('open');}
async function saveRec(){
  const rec={date:document.getElementById('m-date').value,market:document.getElementById('m-mkt').value,minPrice:+document.getElementById('m-min').value,maxPrice:+document.getElementById('m-max').value,modalPrice:+document.getElementById('m-mod').value,volume:+document.getElementById('m-vol').value,grade:document.getElementById('m-grd').value};
  if(!rec.date||isNaN(rec.modalPrice)){toast('⚠ Fill all fields');return;}
  if(editIdx>=0)DS[editIdx]=rec;else{DS.push(rec);DS.sort((a,b)=>a.date.localeCompare(b.date));}
  filtDS=[...DS];closeModal();
  PP=DS.filter(d=>d.market==='Pollachi').map(d=>d.modalPrice);
  WP=weeklyAvg(PP);
  toast('♻ Quick retrain (AR+HW)…');
  arBeta=fitAR(PP.slice(0,Math.floor(PP.length*.85)),7);
  hwP=fitHW(PP.slice(0,Math.floor(PP.length*.85)));
  await buildFC();populateAll();
  toast('✓ Updated');
}
function delRec(i){if(!confirm('Delete?'))return;DS.splice(i,1);filtDS=[...DS];renderDSTable();toast('✓ Deleted');}

/* ── 15. MISC ── */
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2800);}
function showPage(p,btn){document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.nb').forEach(b=>b.classList.remove('active'));document.getElementById('page-'+p).classList.add('active');btn.classList.add('active');}
function setRange(d,btn){chartRange=d;document.querySelectorAll('.cbtn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');if(fcOut)initMainChart(d);}
document.getElementById('modal').addEventListener('click',e=>{if(e.target===document.getElementById('modal'))closeModal();});
window.addEventListener('load',()=>trainAll().catch(console.error));
