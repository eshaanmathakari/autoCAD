"""
Lightweight DXF viewer for Streamlit — self-contained, no src/ dependency.
Uses Three.js + dxf-parser inlined from vendor files (no CDN needed).
"""

import base64

from poc.viewer_deps import get_three_js, get_dxf_parser_js


def generate_viewer_html(dxf_bytes: bytes, height: int = 500) -> str:
    dxf_b64 = base64.b64encode(dxf_bytes).decode("utf-8")
    three_js = get_three_js()
    dxf_parser_js = get_dxf_parser_js()

    return f'''<!DOCTYPE html>
<html>
<head>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{overflow:hidden;font-family:sans-serif}}
#c{{width:100%;height:{height}px;background:#1a1a2e;position:relative}}
#ctrls{{position:absolute;top:8px;left:8px;z-index:100;display:flex;gap:4px}}
#ctrls button{{padding:5px 10px;background:rgba(74,74,106,.9);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px}}
#ctrls button:hover{{background:rgba(106,106,138,.9)}}
#ctrls button.on{{background:rgba(0,150,136,.9)}}
#info{{position:absolute;bottom:8px;left:8px;color:#888;font:11px monospace;background:rgba(26,26,46,.8);padding:3px 6px;border-radius:3px}}
#no-geom{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#ff6b6b;font-size:14px;text-align:center;display:none}}
</style>
<script>{three_js}</script>
<script>{dxf_parser_js}</script>
</head>
<body>
<div id="c">
 <div id="ctrls">
  <button onclick="fit()">Fit</button>
  <button onclick="resetV()">Reset</button>
  <button id="gb" class="on" onclick="togGrid()">Grid</button>
 </div>
 <div id="info">Loading...</div>
 <div id="no-geom">No geometry found in DXF</div>
</div>
<script>
const LC={{"POOL_OUTLINE":0x00ff88,"POOL_STAIRS":0x00ff00,"POOL_DIMENSIONS":0xff4444,"POOL_TEXT":0x00ffff,"POOL_BENCH":0xffff00,"POOL_EQUIPMENT":0x888888,"GEOMETRY":0x00ff88,"TEXT":0x88ff00,"DIMENSIONS":0xff4444,"0":0xffffff}};
const b64="{dxf_b64}";
let dxf,scene,cam,ren,grid,gVis=true,eCnt=0;
let bnd={{x0:Infinity,x1:-Infinity,y0:Infinity,y1:-Infinity}};
function ub(vs){{vs.forEach(v=>{{if(v.x!==undefined){{bnd.x0=Math.min(bnd.x0,v.x);bnd.x1=Math.max(bnd.x1,v.x);bnd.y0=Math.min(bnd.y0,v.y);bnd.y1=Math.max(bnd.y1,v.y);}}}})}}
function lc(n){{return LC[n]||0x00ff88}}
function init(){{
try{{dxf=new DxfParser().parseSync(atob(b64))}}catch(e){{document.getElementById("info").textContent="Parse error: "+e.message;return}}
const c=document.getElementById("c");
scene=new THREE.Scene();scene.background=new THREE.Color(0x1a1a2e);
const a=c.clientWidth/c.clientHeight;
cam=new THREE.OrthographicCamera(-500*a,500*a,500,-500,.1,1000);cam.position.z=100;
ren=new THREE.WebGLRenderer({{antialias:true}});ren.setSize(c.clientWidth,c.clientHeight);ren.setPixelRatio(devicePixelRatio);c.appendChild(ren.domElement);
if(dxf.entities)dxf.entities.forEach(e=>{{
const col=lc(e.layer||"0"),mat=new THREE.LineBasicMaterial({{color:col}});let m=null;
if(e.type==="LINE"&&e.vertices&&e.vertices.length>=2){{const p=[new THREE.Vector3(e.vertices[0].x,e.vertices[0].y,0),new THREE.Vector3(e.vertices[1].x,e.vertices[1].y,0)];m=new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),mat);ub(e.vertices)}}
else if(e.type==="CIRCLE"&&e.center){{const g=new THREE.CircleGeometry(e.radius,48);m=new THREE.LineSegments(new THREE.EdgesGeometry(g),mat);m.position.set(e.center.x,e.center.y,0);ub([{{x:e.center.x-e.radius,y:e.center.y-e.radius}},{{x:e.center.x+e.radius,y:e.center.y+e.radius}}])}}
else if((e.type==="LWPOLYLINE"||e.type==="POLYLINE")&&e.vertices&&e.vertices.length>=2){{const p=e.vertices.map(v=>new THREE.Vector3(v.x,v.y,0));if(e.shape||e.closed)p.push(p[0].clone());m=new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),mat);ub(e.vertices)}}
else if((e.type==="TEXT"||e.type==="MTEXT")&&(e.startPoint||e.position)){{const pos=e.startPoint||e.position;const h=e.textHeight||10,w=(e.text?.length||5)*h*.6;const p=[new THREE.Vector3(pos.x,pos.y,0),new THREE.Vector3(pos.x+w,pos.y,0),new THREE.Vector3(pos.x+w,pos.y+h,0),new THREE.Vector3(pos.x,pos.y+h,0),new THREE.Vector3(pos.x,pos.y,0)];m=new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),new THREE.LineDashedMaterial({{color:col,dashSize:3,gapSize:2}}));m.computeLineDistances();ub([pos,{{x:pos.x+w,y:pos.y+h}}])}}
if(m){{scene.add(m);eCnt++}}
}});
if(eCnt===0){{document.getElementById("no-geom").style.display="block"}}
const gs=Math.max(bnd.x1-bnd.x0,bnd.y1-bnd.y0)*2||1000;
grid=new THREE.GridHelper(gs,20,0x444466,0x333355);grid.rotation.x=Math.PI/2;
grid.position.set((bnd.x0+bnd.x1)/2,(bnd.y0+bnd.y1)/2,-1);scene.add(grid);
document.getElementById("info").textContent=`Entities: ${{eCnt}}`;
fit();setupCtrl();(function a(){{requestAnimationFrame(a);ren.render(scene,cam)}})()}}
function fit(){{if(bnd.x0===Infinity)return;const c=document.getElementById("c"),a=c.clientWidth/c.clientHeight,cx=(bnd.x0+bnd.x1)/2,cy=(bnd.y0+bnd.y1)/2,w=bnd.x1-bnd.x0,h=bnd.y1-bnd.y0;let vw,vh;if(w/h>a){{vw=w*1.15;vh=vw/a}}else{{vh=h*1.15;vw=vh*a}}cam.left=cx-vw/2;cam.right=cx+vw/2;cam.top=cy+vh/2;cam.bottom=cy-vh/2;cam.updateProjectionMatrix()}}
function resetV(){{const c=document.getElementById("c"),a=c.clientWidth/c.clientHeight;cam.left=-500*a;cam.right=500*a;cam.top=500;cam.bottom=-500;cam.updateProjectionMatrix()}}
function togGrid(){{gVis=!gVis;grid.visible=gVis;document.getElementById("gb").classList.toggle("on",gVis)}}
function setupCtrl(){{const cv=ren.domElement;let dr=false,lx,ly;
cv.onmousedown=e=>{{dr=true;lx=e.clientX;ly=e.clientY;cv.style.cursor="grabbing"}};
cv.onmousemove=e=>{{if(!dr)return;const s=(cam.right-cam.left)/cv.clientWidth;cam.left-=(e.clientX-lx)*s;cam.right-=(e.clientX-lx)*s;cam.top+=(e.clientY-ly)*s;cam.bottom+=(e.clientY-ly)*s;cam.updateProjectionMatrix();lx=e.clientX;ly=e.clientY}};
cv.onmouseup=()=>{{dr=false;cv.style.cursor="grab"}};cv.onmouseleave=()=>{{dr=false}};cv.onmouseenter=()=>{{cv.style.cursor="grab"}};
cv.addEventListener("wheel",e=>{{e.preventDefault();const f=e.deltaY>0?1.1:.9,r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;const wx=cam.left+(mx/cv.clientWidth)*(cam.right-cam.left),wy=cam.top-(my/cv.clientHeight)*(cam.top-cam.bottom);const nw=(cam.right-cam.left)*f,nh=(cam.top-cam.bottom)*f;const rx=(wx-cam.left)/(cam.right-cam.left),ry=(cam.top-wy)/(cam.top-cam.bottom);cam.left=wx-rx*nw;cam.right=cam.left+nw;cam.top=wy+ry*nh;cam.bottom=cam.top-nh;cam.updateProjectionMatrix()}},{{passive:false}})}}
init();
</script></body></html>'''
