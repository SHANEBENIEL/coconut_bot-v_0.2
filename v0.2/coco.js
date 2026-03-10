
// ══════════════════════════════════════════
// CACHE SYSTEM
// ══════════════════════════════════════════
const CACHE_KEY='coconutAI_models_v3';

async function loadCachedModels(){
    try{
        const cached=localStorage.getItem(CACHE_KEY);
        if(cached){
            const data=JSON.parse(cached);
            if(Date.now()-data.ts<86400000){  // 24hr cache
                arBeta=data.arBeta; 
                hwP=data.hwP; 
                WT=data.WT;
                CI=data.CI; 
                metrics=data.metrics;
                return true;
            }
        }
    }catch(e){console.error('Cache load error:',e);}
    return false;
}

function cacheModels(){
    try{
        localStorage.setItem(CACHE_KEY,JSON.stringify({
            ts:Date.now(),arBeta,hwP,WT,CI,metrics
        }));
        console.log('✓ Models cached');
    }catch(e){console.error('Cache save error:',e);}
}
// ══════════════════════════════════════════
// 1. DATA GENERATION
// ══════════════════════════════════════════
function generateDataset(){
const mkt=['Pollachi','Coimbatore','Chennai','Tiruppur','Erode','Salem','Madurai'];
const adj={Pollachi:0,Coimbatore:2,Chennai:5,Tiruppur:1,Erode:1.5,Salem:3,Madurai:4};
const grd=['Premium','Grade A','Grade A','Grade B','Mixed'];
const sIdx=[7,5,2,-1,-4,-6,-5,-3,0,3,6,8];
const data=[];
const st={};
mkt.forEach(m=>st[m]=42+adj[m]);
const cur=new Date('2020-01-01');
const end=new Date('2026-02-23');
while(cur<=end){
const ds=cur.toISOString().slice(0,10);
const yr=cur.getFullYear(),mo=cur.getMonth(),dow=cur.getDay();
const today=dow===0?['Pollachi']:[3,6].includes(dow)?mkt.slice(0,4):mkt;
for(const m of today){
const target=44+adj[m]+(yr-2019)*1.8+sIdx[mo];
const mr=(target-st[m])*0.10;
const shock=(Math.random()-0.49)*3.2;
st[m]=Math.max(28,Math.min(115,st[m]+mr+shock));
const modal=Math.round(st[m]*2)/2;
const sp=3.5+Math.random()*7;
const minP=Math.max(24,Math.round((modal-sp*.55)*2)/2);
const maxP=Math.round((modal+sp*.45)*2)/2;
const bv={Pollachi:450,Coimbatore:320,Chennai:580,Tiruppur:180,Erode:200,Salem:150,Madurai:250}[m]||200;
const vol=Math.max(50,Math.round(bv*(0.7+Math.random()*0.6)));
data.push({date:ds,market:m,minPrice:minP,modalPrice:modal,maxPrice:maxP,volume:vol,grade:grd[Math.floor(Math.random()*grd.length)]});
}
cur.setDate(cur.getDate()+1);
}
data.sort((a,b)=>a.date.localeCompare(b.date));
return data;
}
// ══════════════════════════════════════════
// 2. AR(p) MODEL
// ══════════════════════════════════════════
function fitAR(prices,p=7){
const X=[],y=[];
for(let i=p;i<prices.length;i++){
const row=[1];
for(let j=1;j<=p;j++) row.push(prices[i-j]);
X.push(row); y.push(prices[i]);
}
const Xt=transpose(X);
const beta=solveLinear(matMul(Xt,X), matVecMul(Xt,y));
return beta;
}
function arForecast(history,beta,steps){
const p=beta.length-1, buf=[...history.slice(-p)], out=[];
for(let s=0;s<steps;s++){
let v=beta[0];
for(let j=1;j<=p;j++) v+=beta[j]*buf[buf.length-j];
v=Math.max(24,Math.min(120,v));
out.push(v); buf.push(v);
}
return out;
}
// ══════════════════════════════════════════
// 3. HOLT-WINTERS
// ══════════════════════════════════════════
function hwSSE(p,a,b){
let L=p[0],T=p[1]-p[0],sse=0;
for(let i=1;i<p.length;i++){
const pred=L+T, Lp=L, Tp=T;
L=a*p[i]+(1-a)*(Lp+Tp);
T=b*(L-Lp)+(1-b)*Tp;
sse+=(p[i]-pred)**2;
}
return sse;
}
function fitHW(p){
let ba=0.3,bb=0.1,best=Infinity;
for(let a=0.1;a<=0.9;a+=0.2) for(let b=0.1;b<=0.4;b+=0.15){
const e=hwSSE(p,a,b);
if(e<best){best=e;ba=a;bb=b;}
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
// ══════════════════════════════════════════
// 4. LSTM (Optimized - 15 epochs, smaller)
// ══════════════════════════════════════════
const LB=14, EPOCHS=15;
function normArr(arr){const mn=Math.min(...arr),mx=Math.max(...arr),rng=mx-mn||1;return{n:arr.map(v=>(v-mn)/rng),mn,rng};}
function denorm(v,mn,rng){return v*rng+mn;}
function buildXY(norm,lb){
const Xs=[],ys=[];
for(let i=lb;i<norm.length;i++){Xs.push(norm.slice(i-lb,i).map(v=>[v]));ys.push(norm[i]);}
return{Xs,ys};
}
async function trainLSTM(prices,onEpoch){
const{n,mn,rng}=normArr(prices);
const{Xs,ys}=buildXY(n,LB);
const sp=Math.floor(Xs.length*.85);
const xTr=tf.tensor3d(Xs.slice(0,sp)), yTr=tf.tensor2d(ys.slice(0,sp).map(v=>[v]));
const m=tf.sequential();
m.add(tf.layers.lstm({units:16,returnSequences:true,inputShape:[LB,1],dropout:0.1}));
m.add(tf.layers.lstm({units:8,returnSequences:false,dropout:0.1}));
m.add(tf.layers.dense({units:1}));
m.compile({optimizer:tf.train.adam(0.005),loss:'meanSquaredError'});
await m.fit(xTr,yTr,{epochs:EPOCHS,batchSize:64,shuffle:true,
callbacks:{onEpochEnd:async(ep,logs)=>{onEpoch(ep,logs.loss);await tf.nextFrame();}}});
xTr.dispose();yTr.dispose();
return{model:m,n,mn,rng};
}
async function lstmForecast(obj,steps){
const buf=[...obj.n.slice(-LB)],out=[];
for(let s=0;s<steps;s++){
const inp=tf.tensor3d([buf.slice(-LB).map(v=>[v])]);
const res=obj.model.predict(inp);
const raw=(await res.data())[0];
inp.dispose();res.dispose();
const v=Math.max(0,Math.min(1,raw));
buf.push(v);
out.push(Math.max(24,Math.min(120,denorm(v,obj.mn,obj.rng))));
}
return out;
}
// ══════════════════════════════════════════
// 5. LINEAR ALGEBRA
// ══════════════════════════════════════════
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
// ══════════════════════════════════════════
// 6. METRICS & ENSEMBLE
// ══════════════════════════════════════════
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
// ══════════════════════════════════════════
// 7. STATE
// ══════════════════════════════════════════
let DS=[],filtDS=[],dsPg=1;const DPS=25;
let PP=[],TR=0;
let arBeta=null,hwP=null,lstmObj=null,WT={ar:.33,hw:.33,lm:.34},CI={q80:3,q95:6};
let vAct=[],vAR=[],vHW=[],vLM=[],vEns=[];
let metrics={};
let fcOut=null,chartRange=90;
let MC=null,FC=null,MoC=null;
let editIdx=-1;
// ══════════════════════════════════════════
// 8. TRAINING UI HELPERS
// ══════════════════════════════════════════
function step(i,state,val=''){
const si=document.getElementById('si'+i),sl=document.getElementById('sl'+i),sv=document.getElementById('sv'+i);
if(state==='run'){si.className='si run';si.textContent='↻';}
else if(state==='done'){si.className='si done';si.textContent='✓';sl.className='sl done';sv.className='sv done';}
if(val)sv.textContent=val;
}
function prog(p){document.getElementById('pf').style.width=p+'%';}
function elog(m){const e=document.getElementById('elog');e.innerHTML+=m+'<br>';e.scrollTop=e.scrollHeight;}
const delay=ms=>new Promise(r=>setTimeout(r,ms));
// ══════════════════════════════════════════
// 9. TRAINING ORCHESTRATION
// ══════════════════════════════════════════
async function trainAll(){
    try{
        const cached=await loadCachedModels();
        
        step(0,'run');
        DS=generateDataset(); 
        filtDS=[...DS];
        PP=DS.filter(d=>d.market==='Pollachi').map(d=>d.modalPrice);
        TR=Math.floor(PP.length*.85);
        const trP=PP.slice(0,TR), vaP=PP.slice(TR);
        step(0,'done',`${DS.length.toLocaleString()} rows · ${PP.length} Pollachi`);
        prog(15); await delay(50);
        
        if(cached && arBeta && hwP){
            // Use cached AR & HW, but retrain LSTM
            console.log('♻ Using cached AR/HW, training LSTM...');
            document.getElementById('mst').textContent=`Cached AR/HW · Training LSTM...`;
            
            step(1,'done',`MAPE ${metrics.ar}% (cached)`);
            step(2,'done',`MAPE ${metrics.hw}% (cached)`);
            prog(45);
        }else{
            // Train AR(7)
            step(1,'run');
            arBeta=fitAR(trP,7);
            const arVal=arForecast(trP,arBeta,vaP.length);
            const arMP=mape(arVal,vaP).toFixed(2);
            step(1,'done',`MAPE ${arMP}%`);
            prog(30); await delay(40);
            
            // Train Holt-Winters
            step(2,'run');
            hwP=fitHW(trP);
            const hwVal=hwForecast(trP,hwP.alpha,hwP.beta,vaP.length);
            const hwMP=mape(hwVal,vaP).toFixed(2);
            step(2,'done',`MAPE ${hwMP}%`);
            prog(45); await delay(40);
        }
        
        // Always train LSTM (can't cache TF.js models)
        step(3,'run');
        let lastL=0;
        lstmObj=await trainLSTM(trP,(ep,loss)=>{
            lastL=loss;
            if(ep%5===0) elog(`LSTM ${ep+1}/${EPOCHS} loss=${loss.toFixed(5)}`);
            prog(45+Math.round((ep/EPOCHS)*35));
        });
        const lmVal2=await lstmForecast(lstmObj,vaP.length);
        const lmMP=mape(lmVal2,vaP).toFixed(2);
        step(3,'done',`MAPE ${lmMP}%`);
        prog(85); await delay(40);
        
        // Ensemble
        step(4,'run');
        const arVal=arForecast(trP,arBeta,vaP.length);
        const hwVal=hwForecast(trP,hwP.alpha,hwP.beta,vaP.length);
        WT=optWeights(arVal,hwVal,lmVal2,vaP);
        const ensVal=ensemble(arVal,hwVal,lmVal2,WT,vaP.length);
        const enMP=mape(ensVal,vaP).toFixed(2);
        const enMA=mae(ensVal,vaP).toFixed(2);
        const resid=vaP.map((v,i)=>v-ensVal[i]);
        CI=residCI(resid);
        metrics={ar:cached?metrics.ar:+arMP,hw:cached?metrics.hw:+hwMP,lm:+lmMP,ens:+enMP,mae:+enMA,acc:(100-+enMP).toFixed(1)};
        vAct=[...vaP]; vAR=[...arVal]; vHW=[...hwVal]; vLM=[...lmVal2]; vEns=[...ensVal];
        step(4,'done',`MAPE ${enMP}%`);
        prog(100);
        
        // Cache AR/HW only (not LSTM)
        cacheModels();
        
        await delay(300);
        document.getElementById('train-overlay').style.display='none';
        document.getElementById('mst').textContent=`Ens MAPE ${enMP}% · AR+HW+LSTM`;
        
        // Build forecasts and render
        await buildFC();
        populateAll();
        
    }catch(err){
        console.error('Training error:',err);
        elog(`❌ Error: ${err.message}`);
        toast('⚠ Training failed - check console');
    }
}
// ══════════════════════════════════════════
// 10. FORWARD FORECAST
// ══════════════════════════════════════════
async function buildFC(){
    try{
        const N=90;
        if(PP.length<LB){
            toast('⚠ Not enough data for LSTM');
            return;
        }
        const ar90=arForecast(PP,arBeta,N);
        const hw90=hwForecast(PP,hwP.alpha,hwP.beta,N);
        
        // Safety check for LSTM
        if(!lstmObj || !lstmObj.model){
            console.warn('⚠ LSTM model not trained, using AR/HW only');
            const lm90=ar90;  // Fallback to AR
            const en90=ensemble(ar90,hw90,lm90,{ar:0.5,hw:0.5,lm:0},N);
            fcOut={ar:ar90,hw:hw90,lm:lm90,en:en90,N};
        }else{
            const lm90=await lstmForecast(lstmObj,N);
            const en90=ensemble(ar90,hw90,lm90,WT,N);
            fcOut={ar:ar90,hw:hw90,lm:lm90,en:en90,N};
        }
    }catch(err){
        console.error('buildFC error:',err);
        toast('⚠ Forecast build failed');
    }
}
function fcAt(day){
const i=Math.min(day-1,fcOut.N-1);
const pt=fcOut.en[i], hf=Math.sqrt(Math.max(1,day));
return{
pt:Math.round(pt*2)/2,
lm:Math.round(fcOut.lm[i]*2)/2,
ar:Math.round(fcOut.ar[i]*2)/2,
mn80:Math.max(24,Math.round((pt-CI.q80*hf)*2)/2),
mx80:Math.round((pt+CI.q80*hf)*2)/2,
mn95:Math.max(22,Math.round((pt-CI.q95*hf)*2)/2),
mx95:Math.round((pt+CI.q95*hf)*2)/2,
cf:Math.max(55,Math.round(92-day*0.28))
};
}
// ══════════════════════════════════════════
// 11. POPULATE UI
// ══════════════════════════════════════════
function populateAll(){
const cp=PP.at(-1), prev=PP.at(-2)||cp;
const chg=cp-prev, chgP=(chg/prev*100);
const r7=PP.slice(-7), r30=PP.slice(-30);
const rsi=computeRSI(PP.slice(-20)), ma7=MA(PP,7), ma30=MA(PP,30), std30=STD(r30);
document.getElementById('tk0').textContent=`₹${cp}`;
document.getElementById('tk0c').textContent=`${chg>=0?'+':''}${chg.toFixed(1)} (${chgP.toFixed(1)}%)`;
document.getElementById('tk0c').className=`tv ${chg>=0?'up':'down'}`;
document.getElementById('tk1').textContent=`₹${Math.max(...r7)}`;
document.getElementById('tk2').textContent=`₹${Math.min(...r7)}`;
document.getElementById('tk3').textContent=`₹${(r30.reduce((s,v)=>s+v,0)/30).toFixed(1)}`;
document.getElementById('tk4').textContent=`±₹${std30.toFixed(1)}`;
document.getElementById('tk5').textContent=rsi.toFixed(0);
const sig=rsi>60&&cp>ma30?'BUY':rsi<40&&cp<ma30?'SELL':'HOLD';
document.getElementById('tk6').textContent=sig;
const t1=fcAt(1),m1=fcAt(30);
document.getElementById('sc0').textContent=`₹${cp}`;
document.getElementById('sc0').className=`cval ${chg>=0?'up':'down'}`;
document.getElementById('sc0s').textContent=`${chg>=0?'▲':'▼'} ₹${Math.abs(chg).toFixed(1)} (${Math.abs(chgP).toFixed(1)}%)`;
document.getElementById('sc1').textContent=`₹${t1.pt}`;
document.getElementById('sc1').className=`cval ${t1.pt>=cp?'up':'down'}`;
document.getElementById('sc1s').textContent=`Range ₹${t1.mn80} – ₹${t1.mx80}`;
document.getElementById('sc2').textContent=`₹${m1.pt}`;
document.getElementById('sc2').className=`cval ${m1.pt>=cp?'up':'down'}`;
document.getElementById('sc2s').textContent=`Trend: ${m1.pt>cp?'↗ Upward':'↘ Downward'}`;
document.getElementById('sc3').textContent=`${metrics.ens}%`;
document.getElementById('sc3s').textContent=`MAE ₹${metrics.mae}`;
renderFCCards(cp);
renderTech(cp,ma7,ma30,rsi,std30,chgP);
renderSeas();
renderFCTable(cp);
renderDSTable();
renderMetrics();
renderFeatBars();
setTimeout(()=>{ initMainChart(chartRange); },100);
setTimeout(()=>{ initFCChart(); initMonthChart(); },300);
setTimeout(()=>{ initResidChart(); initValChart(); initWgtChart(); },600);
}
function renderFCCards(cp){
const pds=[{l:'Tomorrow',d:1,i:'📅'},{l:'Next Week',d:7,i:'📆'},{l:'Next Month',d:30,i:'🗓'},{l:'3 Months',d:90,i:'📊'}];
document.getElementById('fc-cards').innerHTML=pds.map(p=>{
const f=fcAt(p.d),up=f.pt>cp,pct=((f.pt-cp)/cp*100).toFixed(1);
return`<div class="fcc">
<div class="fper">${p.i} ${p.l}</div>
<div class="frng"><span class="down">₹${f.mn80}</span><span style="color:var(--text3);margin:0 4px">—</span><span class="up">₹${f.mx80}</span></div>
<div class="fexp">Expected: <strong style="color:var(--text)">₹${f.pt}/kg</strong></div>
<div class="crow"><span>${f.cf}%</span><div class="cbar"><div class="cfil" style="width:${f.cf}%"></div></div></div>
<div class="ftr ${up?'up':'down'}">${up?'▲':'▼'} ${Math.abs(pct)}%</div>
<div class="mbdg">AR+HW+LSTM</div>
</div>`;
}).join('');
}
function renderTech(cp,ma7,ma30,rsi,std30,chgP){
const mom=((cp-(PP.at(-8)||cp))/(PP.at(-8)||cp)*100);
const rows=[
['MA-7',`₹${ma7.toFixed(1)}`,cp>ma7?'bb':'bd',cp>ma7?'BULLISH':'BEARISH'],
['MA-30',`₹${ma30.toFixed(1)}`,cp>ma30?'bb':'bd',cp>ma30?'BULLISH':'BEARISH'],
['RSI(14)',rsi.toFixed(0),rsi>70?'bd':rsi<30?'bb':'bn',rsi>70?'OVERBOUGHT':rsi<30?'OVERSOLD':'NEUTRAL'],
['Volatility σ',`₹${std30.toFixed(2)}`,'bn','MODERATE'],
['7D Momentum',`${mom.toFixed(1)}%`,mom>0?'bb':'bd',mom>0?'POSITIVE':'NEGATIVE'],
['Daily Δ',`${chgP.toFixed(1)}%`,chgP>=0?'bb':'bd',chgP>=0?'UP':'DOWN'],
];
document.getElementById('tech-sigs').innerHTML=rows.map(r=>
`<div class="si-item"><span class="sn">${r[0]}</span>
<div style="display:flex;align-items:center;gap:6px"><span class="sv2">${r[1]}</span><span class="badge ${r[2]}">${r[3]}</span></div></div>`
).join('');
}
function renderSeas(){
const mo=new Date('2026-02-23').getMonth();
const seas=['Winter','Late Winter','Spring','Pre-Summer','Summer','Monsoon','Monsoon','Monsoon','Post-Mon','Festive','NE Monsoon','Peak Winter'];
const harv=['Post-Harvest','Post-Harvest','Mid Season','Off Season','Off Season','Pre-Harvest','Harvest','Harvest','Post-Harvest','Festival','Festival','Pre-Season'];
const rows=[
['Season',seas[mo]],['Harvest Phase',harv[mo]],
['AR(7) Tomorrow',`₹${arForecast(PP,arBeta,1)[0].toFixed(1)}`],
['HW Tomorrow',`₹${hwForecast(PP,hwP.alpha,hwP.beta,1)[0].toFixed(1)}`],
['LSTM Tomorrow',`₹${fcOut.lm[0].toFixed(1)}`],
['Ensemble Tomorrow',`₹${fcOut.en[0].toFixed(1)}`],
];
document.getElementById('seas-sigs').innerHTML=rows.map(r=>
`<div class="si-item"><span class="sn">${r[0]}</span><span class="sv2">${r[1]}</span></div>`
).join('');
}
function renderFCTable(cp){
const today=new Date('2026-02-23');
document.getElementById('fc-tbody').innerHTML=Array.from({length:12},(_,i)=>{
const d=(i+1)*7, f=fcAt(d);
const s=new Date(today);s.setDate(s.getDate()+i*7+1);
const e=new Date(today);e.setDate(e.getDate()+(i+1)*7);
const pr=`${s.toLocaleDateString('en-IN',{month:'short',day:'2-digit'})} – ${e.toLocaleDateString('en-IN',{month:'short',day:'2-digit'})}`;
const ch=((f.pt-cp)/cp*100);
const arUp=fcOut.ar[d-1]>cp, lmUp=fcOut.lm[d-1]>cp;
return`<tr>
<td style="font-family:'Instrument Sans',sans-serif">W${i+1}</td>
<td style="font-family:'Instrument Sans',sans-serif;color:var(--text2)">${pr}</td>
<td class="down">₹${f.mn80}</td>
<td style="color:var(--text);font-weight:600">₹${f.pt}</td>
<td class="up">₹${f.mx80}</td>
<td class="${ch>=0?'up':'down'}">${ch>=0?'+':''}${ch.toFixed(1)}%</td>
<td><div style="display:flex;align-items:center;gap:5px"><div class="cbar" style="width:48px"><div class="cfil" style="width:${f.cf}%"></div></div><span style="font-size:.65rem;color:var(--text2)">${f.cf}%</span></div></td>
<td><span class="badge ${arUp?'bb':'bd'}">${arUp?'↑':'↓'} AR</span></td>
<td><span class="badge ${lmUp?'bb':'bd'}">${lmUp?'↑':'↓'} LSTM</span></td>
</tr>`;
}).join('');
}
function renderMetrics(){
document.getElementById('mgrid').innerHTML=[
{n:`${metrics.ens}%`,l:'Ensemble MAPE'},{n:`₹${metrics.mae}`,l:'Ensemble MAE'},
{n:`${metrics.ar}%`,l:'AR(7) MAPE'},{n:`${metrics.lm}%`,l:'LSTM MAPE'},
].map(m=>`<div class="mbox"><div class="mnum">${m.n}</div><div class="mlbl">${m.l}</div></div>`).join('');
}
function renderFeatBars(){
const coefs=arBeta.slice(1).map((v,i)=>({n:`Lag ${i+1}`,v:Math.abs(v)}));
const mx=Math.max(...coefs.map(c=>c.v));
document.getElementById('feat-bars').innerHTML=coefs.map(c=>
`<div class="fbr"><div class="fn">${c.n}</div>
<div class="fb"><div class="ff" style="width:${(c.v/mx*100).toFixed(0)}%"></div></div>
<div class="fp">${(c.v/mx*100).toFixed(0)}%</div></div>`
).join('');
}
// ══════════════════════════════════════════
// 12. CHARTS (🆕 FIX: Added aspectRatio)
// ══════════════════════════════════════════
const CO={responsive:true,maintainAspectRatio:false,aspectRatio:2,
interaction:{mode:'index',intersect:false},
plugins:{legend:{display:false},tooltip:{backgroundColor:'#0e0e12',borderColor:'#252535',borderWidth:1,titleColor:'#e8edf5',bodyColor:'#7a8494',callbacks:{label:c=>`${c.dataset.label}: ₹${c.raw??'N/A'}`}}},
scales:{x:{grid:{color:'#1c1c24'},ticks:{color:'#3a4050',font:{family:'JetBrains Mono',size:9},maxTicksLimit:10}},
y:{grid:{color:'#1c1c24'},ticks:{color:'#3a4050',font:{family:'JetBrains Mono',size:9},callback:v=>`₹${v}`}}}};
function initMainChart(days){
    try{
        const ctx=document.getElementById('mainChart').getContext('2d');
        if(MC)MC.destroy();
        if(!fcOut || !fcOut.en.length){
            console.warn('No forecast data for chart');
            return;
        }
        const hist=PP.slice(-Math.min(days,PP.length));
        const hdates=DS.filter(d=>d.market==='Pollachi').map(d=>d.date).slice(-hist.length);
        const FD=30;const today=new Date('2026-02-23');
        const fl=Array.from({length:FD},(_,i)=>{const d=new Date(today);d.setDate(d.getDate()+i+1);return d.toISOString().slice(0,10);});
        const al=[...hdates,...fl];
        const ah=[...hist,...Array(FD).fill(null)];
        const ef=[...Array(hist.length-1).fill(null),hist.at(-1),...fcOut.en.slice(0,FD)];
        const lf=[...Array(hist.length-1).fill(null),hist.at(-1),...fcOut.lm.slice(0,FD)];
        const cimin=[...Array(hist.length).fill(null),...fcOut.en.slice(0,FD).map((v,i)=>v-CI.q80*Math.sqrt(i+1))];
        const cimax=[...Array(hist.length).fill(null),...fcOut.en.slice(0,FD).map((v,i)=>v+CI.q80*Math.sqrt(i+1))];
        MC=new Chart(ctx,{type:'line',data:{labels:al,datasets:[
        {label:'CI Min',data:cimin,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.07)',fill:'+1',pointRadius:0},
        {label:'CI Max',data:cimax,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.07)',fill:'-1',pointRadius:0},
        {label:'Actual',data:ah,borderColor:'#10b981',borderWidth:1.8,pointRadius:0,tension:.3},
        {label:'Ensemble',data:ef,borderColor:'#818cf8',borderWidth:2,borderDash:[6,3],pointRadius:0,tension:.4},
        {label:'LSTM',data:lf,borderColor:'#22d3ee',borderWidth:1.4,borderDash:[3,3],pointRadius:0,tension:.4},
        ]},options:{...CO}});
    }catch(err){
        console.error('initMainChart error:',err);
    }
}
function initFCChart(){
    try{
        const ctx=document.getElementById('fcastChart').getContext('2d');
        if(FC)FC.destroy();
        if(!fcOut || !fcOut.en.length){
            console.warn('No forecast data for chart');
            return;
        }    
        const today=new Date('2026-02-23');
        const labels=Array.from({length:fcOut.N},(_,i)=>{const d=new Date(today);d.setDate(d.getDate()+i+1);return d.toISOString().slice(0,10);});
        const ci80mn=fcOut.en.map((v,i)=>v-CI.q80*Math.sqrt(i+1));
        const ci80mx=fcOut.en.map((v,i)=>v+CI.q80*Math.sqrt(i+1));
        const ci95mn=fcOut.en.map((v,i)=>v-CI.q95*Math.sqrt(i+1));
        const ci95mx=fcOut.en.map((v,i)=>v+CI.q95*Math.sqrt(i+1));
        FC=new Chart(ctx,{type:'line',data:{labels,datasets:[
        {label:'95% CI Min',data:ci95mn,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.04)',fill:'+1',pointRadius:0},
        {label:'95% CI Max',data:ci95mx,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.04)',fill:'-1',pointRadius:0},
        {label:'80% CI Min',data:ci80mn,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.11)',fill:'+1',pointRadius:0},
        {label:'80% CI Max',data:ci80mx,borderColor:'transparent',backgroundColor:'rgba(129,140,248,.11)',fill:'-1',pointRadius:0},
        {label:'Ensemble',data:fcOut.en,borderColor:'#818cf8',borderWidth:2.5,pointRadius:0,tension:.4},
        {label:'LSTM',data:fcOut.lm,borderColor:'#22d3ee',borderWidth:1.5,borderDash:[4,2],pointRadius:0,tension:.4},
        {label:'AR(7)',data:fcOut.ar,borderColor:'#f97316',borderWidth:1.2,borderDash:[2,4],pointRadius:0,tension:.3},
        {label:'HW',data:fcOut.hw,borderColor:'#f59e0b',borderWidth:1,borderDash:[1,4],pointRadius:0,tension:.3},
        ]},options:{...CO,plugins:{...CO.plugins,legend:{display:true,labels:{color:'#7a8494',font:{family:'Instrument Sans',size:10}}}}}});
    }catch(err){
        console.error('initFCChart error:',err);
    }    
}
function initMonthChart(){
const ctx=document.getElementById('monthChart');if(!ctx)return;
const today=new Date('2026-02-23');
const labels=Array.from({length:6},(_,i)=>{const d=new Date(today);d.setMonth(d.getMonth()+i+1);return d.toLocaleDateString('en-IN',{month:'short',year:'2-digit'});});
const md=Array.from({length:6},(_,i)=>{const f=fcAt((i+1)*30);return{mn:f.mn80,ex:f.pt,mx:f.mx80};});
if(MoC)MoC.destroy();
MoC=new Chart(ctx,{type:'bar',data:{labels,datasets:[
{label:'Min',data:md.map(d=>d.mn),backgroundColor:'rgba(239,68,68,.28)',borderColor:'rgba(239,68,68,.7)',borderWidth:1,borderRadius:4},
{label:'Expected',data:md.map(d=>d.ex),backgroundColor:'rgba(34,211,238,.32)',borderColor:'rgba(34,211,238,.8)',borderWidth:1,borderRadius:4},
{label:'Max',data:md.map(d=>d.mx),backgroundColor:'rgba(129,140,248,.28)',borderColor:'rgba(129,140,248,.7)',borderWidth:1,borderRadius:4},
]},options:{...CO,plugins:{...CO.plugins,legend:{display:true,labels:{color:'#7a8494',font:{family:'Instrument Sans',size:10}}}}}});
}
function initResidChart(){
const ctx=document.getElementById('residChart').getContext('2d');
const r=vAct.map((v,i)=>v-vEns[i]);
new Chart(ctx,{type:'bar',data:{labels:r.map((_,i)=>`v${i+1}`),datasets:[{label:'Residual',data:r,
backgroundColor:r.map(v=>v>=0?'rgba(16,185,129,.45)':'rgba(239,68,68,.45)'),
borderColor:r.map(v=>v>=0?'rgba(16,185,129,.8)':'rgba(239,68,68,.8)'),borderWidth:1,borderRadius:2}]},
options:{...CO,plugins:{...CO.plugins,legend:{display:false}}}});
}
function initValChart(){
const ctx=document.getElementById('valChart').getContext('2d');
const l=vAct.map((_,i)=>`d${i+1}`);
new Chart(ctx,{type:'line',data:{labels:l,datasets:[
{label:'Actual',data:vAct,borderColor:'#10b981',borderWidth:2,pointRadius:0,tension:.3},
{label:'Ensemble',data:vEns,borderColor:'#818cf8',borderWidth:1.5,borderDash:[4,2],pointRadius:0,tension:.3},
{label:'AR(7)',data:vAR,borderColor:'#f97316',borderWidth:1.1,borderDash:[2,3],pointRadius:0,tension:.3},
{label:'LSTM',data:vLM,borderColor:'#22d3ee',borderWidth:1.1,borderDash:[3,2],pointRadius:0,tension:.3},
{label:'HW',data:vHW,borderColor:'#f59e0b',borderWidth:1,borderDash:[1,4],pointRadius:0,tension:.3},
]},options:{...CO,plugins:{...CO.plugins,legend:{display:true,labels:{color:'#7a8494',font:{family:'Instrument Sans',size:10}}}}}});
}
function initWgtChart(){
const ctx=document.getElementById('wgtChart').getContext('2d');
new Chart(ctx,{type:'doughnut',data:{
labels:[`AR(7) ${(WT.ar*100).toFixed(0)}%`,`HW ${(WT.hw*100).toFixed(0)}%`,`LSTM ${(WT.lm*100).toFixed(0)}%`],
datasets:[{data:[WT.ar,WT.hw,WT.lm],backgroundColor:['#f97316','#f59e0b','#22d3ee'],borderColor:'#09090b',borderWidth:3}]
},options:{responsive:true,maintainAspectRatio:false,aspectRatio:1.5,plugins:{legend:{position:'bottom',labels:{color:'#7a8494',font:{family:'Instrument Sans',size:11},padding:10}},tooltip:{backgroundColor:'#0e0e12',borderColor:'#252535',borderWidth:1}}}});
}
// ══════════════════════════════════════════
// 13. DATASET TABLE
// ══════════════════════════════════════════
function renderDSTable(){
const s=(dsPg-1)*DPS, sl=filtDS.slice(s,s+DPS);
document.getElementById('ds-cnt').textContent=`${DS.length.toLocaleString()} total · ${PP.length} Pollachi`;
document.getElementById('ds-info').textContent=`${s+1}–${Math.min(s+DPS,filtDS.length)} of ${filtDS.length.toLocaleString()}`;
document.getElementById('ds-pg').textContent=`${dsPg}/${Math.ceil(filtDS.length/DPS)}`;
document.getElementById('ds-tbody').innerHTML=sl.map((r,i)=>{
const gi=DS.indexOf(r);
return`<tr>
<td style="color:var(--text3)">${s+i+1}</td>
<td style="font-family:'Instrument Sans',sans-serif">${r.date}</td>
<td style="font-family:'Instrument Sans',sans-serif">${r.market}</td>
<td class="down">₹${r.minPrice}</td>
<td style="font-weight:600">₹${r.modalPrice}</td>
<td class="up">₹${r.maxPrice}</td>
<td>${r.volume}</td>
<td><span class="badge bn" style="font-size:.57rem">${r.grade}</span></td>
<td><div style="display:flex;gap:4px">
<button class="btn" style="padding:2px 8px;font-size:.68rem" onclick="editRec(${gi})">Edit</button>
<button class="btn" style="padding:2px 8px;font-size:.68rem;color:var(--down)" onclick="delRec(${gi})">Del</button>
</div></td>
</tr>`;
}).join('');
}
function filterDS(){
const q=document.getElementById('dss').value.toLowerCase();
const yr=document.getElementById('dsy').value, mk=document.getElementById('dsm').value;
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
// ══════════════════════════════════════════
// 14. MODAL
// ══════════════════════════════════════════
function openAdd(){editIdx=-1;document.getElementById('mtitle').textContent='+ Add Record';['m-min','m-max','m-mod','m-vol'].forEach(id=>document.getElementById(id).value='');document.getElementById('m-date').value='2026-02-24';document.getElementById('modal').classList.add('open');}
function editRec(i){editIdx=i;const r=DS[i];document.getElementById('mtitle').textContent='Edit Record';document.getElementById('m-date').value=r.date;document.getElementById('m-mkt').value=r.market;document.getElementById('m-min').value=r.minPrice;document.getElementById('m-max').value=r.maxPrice;document.getElementById('m-mod').value=r.modalPrice;document.getElementById('m-vol').value=r.volume;document.getElementById('m-grd').value=r.grade;document.getElementById('modal').classList.add('open');}
function closeModal(){document.getElementById('modal').classList.remove('open');}
async function saveRec(){
const rec={date:document.getElementById('m-date').value,market:document.getElementById('m-mkt').value,
minPrice:+document.getElementById('m-min').value,maxPrice:+document.getElementById('m-max').value,
modalPrice:+document.getElementById('m-mod').value,volume:+document.getElementById('m-vol').value,
grade:document.getElementById('m-grd').value};
if(!rec.date||isNaN(rec.modalPrice)){toast('⚠ Fill all fields');return;}
if(editIdx>=0)DS[editIdx]=rec;else{DS.push(rec);DS.sort((a,b)=>a.date.localeCompare(b.date));}
filtDS=[...DS];closeModal();
PP=DS.filter(d=>d.market==='Pollachi').map(d=>d.modalPrice);
toast('♻ Quick retrain (AR+HW only)…');
arBeta=fitAR(PP.slice(0,Math.floor(PP.length*.85)),7);
hwP=fitHW(PP.slice(0,Math.floor(PP.length*.85)));
await buildFC();populateAll();
toast('✓ Models updated');
}
function delRec(i){if(!confirm('Delete?'))return;DS.splice(i,1);filtDS=[...DS];renderDSTable();toast('✓ Deleted');}
// ══════════════════════════════════════════
// 15. MISC
// ══════════════════════════════════════════
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2800);}
function showPage(p,btn){document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.nb').forEach(b=>b.classList.remove('active'));document.getElementById('page-'+p).classList.add('active');btn.classList.add('active');}
function setRange(d,btn){chartRange=d;document.querySelectorAll('.cbtn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');if(fcOut)initMainChart(d);}
document.getElementById('modal').addEventListener('click',e=>{if(e.target===document.getElementById('modal'))closeModal();});
window.addEventListener('load',()=>trainAll().catch(console.error));
