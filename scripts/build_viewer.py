"""build_viewer.py - 설치 없이 브라우저에서 여는 3-way 모션 비교 뷰어(visualization/motion_viewer.html) 생성.
기능: Raw | Basic | Enhanced 동시 재생, Play/Pause, 프레임 슬라이더, 속도, 마우스 드래그 회전, 지면 grid,
      문제 관절 강조(빨강=지면 아래, 주황 원=spike), 프레임별·영상별 지표 표시.  데이터는 HTML 안에 포함된다."""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import motion_lib as ml

NAMES = {"idle": "Idle", "sidekick_mid": "Kick A", "sidekick_high": "Kick B", "roll": "Roll", "handstand": "Handstand"}
mt = pd.read_csv(os.path.join(ROOT, "results", "methods_per_video.csv"))
data = {}
for v, lab in NAMES.items():
    seq = {k: np.load(os.path.join(ROOT, "data", "processed", f"{v}_{t}.npy")) for k, t in (("raw", "raw"), ("basic", "target"), ("enh", "enhanced"))}
    fps = float(np.load(os.path.join(ROOT, "data", "raw_keypoints", f"{v}.npz"))["fps"])
    m = mt[mt.video == v]
    summ = {k: {c: round(float(m[m.method.str.startswith(p)][c].iloc[0]), 2) for c in ("jerk_m_s3", "bone_cv_pct", "root_y_jitter_m_s2", "penetration_pct", "n_spikes")}
            for k, p in (("raw", "E0"), ("basic", "E1"), ("enh", "E6"))}
    data[v] = {"label": lab, "fps": fps, "summary": summ, "spikes": ml.spike_mask(seq["raw"]).any(axis=2).astype(int).tolist(),
               **{k: np.round(s * 1000).astype(int).reshape(len(s), -1).tolist() for k, s in seq.items()}}   # mm 정수
HTML = r"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>3D Motion Post-Processing Viewer</title>
<style>body{margin:0;font-family:'Malgun Gothic','Noto Sans KR',sans-serif;background:#f4f5f7;color:#222}
#top{padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:#fff;border-bottom:1px solid #ddd}
button,select{font-size:15px;padding:5px 12px} #f{flex:1;min-width:240px} canvas{display:block;width:100%;background:#fafafa}
table{border-collapse:collapse;font-size:13px;margin:8px 16px} td,th{border:1px solid #ccc;padding:3px 10px;text-align:right} th{background:#eee}
.raw{color:#777}.basic{color:#d06f00}.enh{color:#1f6fd0} #note{font-size:12px;color:#555;margin:0 16px 10px}</style></head><body>
<div id="top"><b>3D Motion 비교 뷰어</b><select id="v"></select><button id="p">⏸ Pause</button>
<input type="range" id="f" min="0" value="0"><span id="fl"></span>
<label>속도 <select id="s"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>x</label></div>
<canvas id="c" height="560"></canvas>
<table id="t"></table><div id="note">마우스 드래그: 회전 · 빨간 점: 지면(y=0)보다 1cm 넘게 아래인 관절 · 주황 원: Raw에서 검출된 한 프레임 이상치(spike) · 지표 정의는 README 참고</div>
<script>const D=__DATA__;
const CON=[[11,12],[12,14],[14,16],[11,13],[13,15],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32]];
const BODY=[0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32],KEYS=[['raw','Raw','#888'],['basic','Basic (교수님)','#e07b00'],['enh','Enhanced','#1f6fd0']];
let vid=Object.keys(D)[0],fr=0,play=true,yaw=0.6,pitch=0.18,last=0,acc=0;const c=document.getElementById('c'),g=c.getContext('2d');
const $=id=>document.getElementById(id);for(const k in D){const o=document.createElement('option');o.value=k;o.textContent=D[k].label;$('v').appendChild(o)}
function setVid(k){vid=k;fr=0;$('f').max=D[k].raw.length-1;table()} $('v').onchange=e=>setVid(e.target.value);
$('p').onclick=()=>{play=!play;$('p').textContent=play?'⏸ Pause':'▶ Play'};$('f').oninput=e=>{fr=+e.target.value};
let drag=null;c.onmousedown=e=>drag=[e.clientX,e.clientY];window.onmouseup=()=>drag=null;
window.onmousemove=e=>{if(drag){yaw+=(e.clientX-drag[0])*0.01;pitch=Math.max(-0.3,Math.min(1.2,pitch+(e.clientY-drag[1])*0.005));drag=[e.clientX,e.clientY]}};
function proj(x,y,z,cx,cy,sc){const X=x*Math.cos(yaw)+z*Math.sin(yaw),Z=-x*Math.sin(yaw)+z*Math.cos(yaw);return[cx+X*sc,cy-(y*Math.cos(pitch)-Z*Math.sin(pitch))*sc]}
function table(){const s=D[vid].summary,rows=[['Jerk (m/s³)','jerk_m_s3'],['뼈 길이 CV (%)','bone_cv_pct'],['골반 높이 떨림 (m/s²)','root_y_jitter_m_s2'],['지면 관통 비율 (%)','penetration_pct'],['남은 spike (개)','n_spikes']];
$('t').innerHTML='<tr><th>영상 전체 지표</th><th class=raw>Raw</th><th class=basic>Basic</th><th class=enh>Enhanced</th></tr>'+rows.map(r=>`<tr><td style="text-align:left">${r[0]}</td><td>${s.raw[r[1]]}</td><td>${s.basic[r[1]]}</td><td>${s.enh[r[1]]}</td></tr>`).join('')}
function draw(ts){const dt=(ts-last)/1000;last=ts;const d=D[vid],T=d.raw.length;if(play){acc+=dt*d.fps*(+$('s').value);while(acc>=1){fr=(fr+1)%T;acc-=1}}
c.width=c.clientWidth;const W=c.width,H=c.height,pw=W/3,sc=Math.min(pw*0.36,H*0.26);g.clearRect(0,0,W,H);$('f').value=fr;$('fl').textContent=`frame ${fr+1}/${T} · ${(fr/d.fps).toFixed(2)}s`;
KEYS.forEach(([k,lab,col],i)=>{const cx=pw*(i+0.5),cy=H*0.66,P=d[k][fr];g.strokeStyle='#d5d5d5';g.lineWidth=1;
for(let a=-1;a<=1.001;a+=0.25){let p=proj(a,0,-1,cx,cy,sc),q=proj(a,0,1,cx,cy,sc);g.beginPath();g.moveTo(...p);g.lineTo(...q);g.stroke();p=proj(-1,0,a,cx,cy,sc);q=proj(1,0,a,cx,cy,sc);g.beginPath();g.moveTo(...p);g.lineTo(...q);g.stroke()}
const J=j=>proj(P[j*3]/1000,P[j*3+1]/1000,P[j*3+2]/1000,cx,cy,sc);g.strokeStyle=col;g.lineWidth=4;g.lineCap='round';
CON.forEach(([a,b])=>{g.beginPath();g.moveTo(...J(a));g.lineTo(...J(b));g.stroke()});let nb=0;
BODY.forEach(j=>{const below=P[j*3+1]<-10;if(below)nb++;g.fillStyle=below?'#d62728':col;g.beginPath();g.arc(...J(j),below?6:3.5,0,7);g.fill();
if(k=='raw'&&d.spikes[fr][j]){g.strokeStyle='#ff8c00';g.lineWidth=2;g.beginPath();g.arc(...J(j),11,0,7);g.stroke();g.strokeStyle=col;g.lineWidth=4}});
const hip=(P[23*3+1]+P[24*3+1])/20,th=Math.hypot(P[69]-P[75],P[70]-P[76],P[71]-P[77])/10;
g.fillStyle=col;g.font='bold 18px sans-serif';g.textAlign='center';g.fillText(lab,cx,26);g.font='13px sans-serif';g.fillStyle=nb?'#d62728':'#444';
g.fillText(`지면 아래 관절 ${nb}개 · 골반 높이 ${hip.toFixed(1)}cm · L허벅지 ${th.toFixed(1)}cm`,cx,H-12)});requestAnimationFrame(draw)}
setVid(vid);requestAnimationFrame(draw);</script></body></html>"""
out = os.path.join(ROOT, "visualization", "motion_viewer.html")
open(out, "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print("saved", out, round(os.path.getsize(out) / 1e6, 2), "MB")
