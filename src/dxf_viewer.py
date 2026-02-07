"""
DXF Viewer component for Streamlit using Three.js + dxf-parser.

Provides an interactive in-browser viewer for DXF files with:
- Pan (drag)
- Zoom (scroll)
- Fit to view
- Grid toggle
- Layer visibility controls
"""

import base64
import os
import sys

# Add parent to path so poc.viewer_deps is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from poc.viewer_deps import get_three_js, get_dxf_parser_js


def generate_viewer_html(dxf_bytes: bytes, height: int = 500) -> str:
    """
    Generate HTML with embedded Three.js DXF viewer.

    Args:
        dxf_bytes: Raw DXF file bytes
        height: Viewer height in pixels

    Returns:
        HTML string with embedded viewer
    """
    # Base64 encode DXF for embedding
    dxf_b64 = base64.b64encode(dxf_bytes).decode('utf-8')
    three_js = get_three_js()
    dxf_parser_js = get_dxf_parser_js()

    return f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
        #canvas-container {{
            width: 100%;
            height: {height}px;
            background: #1a1a2e;
            position: relative;
        }}
        #controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            display: flex;
            gap: 4px;
        }}
        #controls button {{
            padding: 6px 12px;
            background: rgba(74, 74, 106, 0.9);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.2s;
        }}
        #controls button:hover {{ background: rgba(106, 106, 138, 0.9); }}
        #controls button.active {{ background: rgba(0, 150, 136, 0.9); }}
        #info {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: #888;
            font-family: monospace;
            font-size: 11px;
            background: rgba(26, 26, 46, 0.8);
            padding: 4px 8px;
            border-radius: 3px;
        }}
        #layer-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(26, 26, 46, 0.95);
            padding: 10px;
            border-radius: 4px;
            color: #ccc;
            font-size: 11px;
            max-height: {height - 40}px;
            overflow-y: auto;
            display: none;
        }}
        #layer-panel.visible {{ display: block; }}
        #layer-panel h4 {{ margin-bottom: 8px; color: #fff; }}
        .layer-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin: 4px 0;
        }}
        .layer-item input {{ cursor: pointer; }}
        .layer-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        #error-msg {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ff6b6b;
            font-size: 14px;
            text-align: center;
            display: none;
        }}
    </style>
    <script>{three_js}</script>
    <script>{dxf_parser_js}</script>
</head>
<body>
    <div id="canvas-container">
        <div id="controls">
            <button onclick="fitToView()" title="Fit drawing to view">Fit</button>
            <button onclick="resetView()" title="Reset to default view">Reset</button>
            <button id="grid-btn" class="active" onclick="toggleGrid()" title="Toggle grid">Grid</button>
            <button id="layers-btn" onclick="toggleLayerPanel()" title="Toggle layer panel">Layers</button>
        </div>
        <div id="layer-panel">
            <h4>Layers</h4>
            <div id="layer-list"></div>
        </div>
        <div id="info">Loading...</div>
        <div id="error-msg"></div>
    </div>

    <script>
        // Layer colors (AutoCAD Color Index approximation)
        const ACI_COLORS = {{
            0: 0xffffff,  // ByBlock
            1: 0xff0000,  // Red
            2: 0xffff00,  // Yellow
            3: 0x00ff00,  // Green
            4: 0x00ffff,  // Cyan
            5: 0x0000ff,  // Blue
            6: 0xff00ff,  // Magenta
            7: 0xffffff,  // White
            8: 0x808080,  // Gray
        }};

        const LAYER_COLORS = {{
            'GEOMETRY': 0x00ff88,
            'TEXT': 0x88ff00,
            'DIMENSIONS': 0xff4444,
            'CENTERLINES': 0x00ffff,
            'HIDDEN': 0x666666,
            'CONSTRUCTION': 0xffff00,
            '0': 0xffffff,
            'POOL_OUTLINE': 0x00ff88,
            'POOL_STAIRS': 0x00ff00,
            'POOL_DIMENSIONS': 0xff4444,
            'POOL_TEXT': 0x00ffff,
            'POOL_BENCH': 0xffff00,
            'POOL_EQUIPMENT': 0x888888,
        }};

        // DXF content (base64 encoded)
        const dxfBase64 = "{dxf_b64}";

        let dxf;
        let scene, camera, renderer;
        let entityCount = 0;
        let bounds = {{ minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }};
        let gridHelper;
        let gridVisible = true;
        let layerGroups = {{}};

        function showError(msg) {{
            document.getElementById('error-msg').style.display = 'block';
            document.getElementById('error-msg').textContent = msg;
            document.getElementById('info').textContent = 'Error';
        }}

        function init() {{
            try {{
                // Decode DXF
                const dxfContent = atob(dxfBase64);

                // Parse DXF
                const parser = new DxfParser();
                dxf = parser.parseSync(dxfContent);
            }} catch(e) {{
                showError('Failed to parse DXF: ' + e.message);
                return;
            }}

            // Three.js setup
            const container = document.getElementById('canvas-container');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            const aspect = container.clientWidth / container.clientHeight;
            camera = new THREE.OrthographicCamera(-500 * aspect, 500 * aspect, 500, -500, 0.1, 1000);
            camera.position.z = 100;

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // Draw entities
            drawEntities();

            // Add grid
            const gridSize = Math.max(
                bounds.maxX - bounds.minX,
                bounds.maxY - bounds.minY
            ) * 2 || 1000;
            gridHelper = new THREE.GridHelper(gridSize, 20, 0x444466, 0x333355);
            gridHelper.rotation.x = Math.PI / 2;
            const cx = (bounds.minX + bounds.maxX) / 2 || 0;
            const cy = (bounds.minY + bounds.maxY) / 2 || 0;
            gridHelper.position.set(cx, cy, -1);
            scene.add(gridHelper);

            // Update info
            document.getElementById('info').textContent =
                `Entities: ${{entityCount}} | Layers: ${{Object.keys(layerGroups).length}}`;

            if (entityCount === 0) {{
                showError('No geometry found in DXF');
            }}

            // Build layer panel
            buildLayerPanel();

            // Initial fit
            fitToView();

            // Setup controls
            setupControls();

            // Start render loop
            animate();
        }}

        function getLayerColor(layerName) {{
            if (LAYER_COLORS[layerName]) return LAYER_COLORS[layerName];
            // Generate consistent color from layer name
            let hash = 0;
            for (let i = 0; i < layerName.length; i++) {{
                hash = layerName.charCodeAt(i) + ((hash << 5) - hash);
            }}
            return (hash & 0x00FFFFFF) | 0x404040;
        }}

        function drawEntities() {{
            if (!dxf.entities) return;

            dxf.entities.forEach(entity => {{
                const layerName = entity.layer || '0';

                // Create layer group if needed
                if (!layerGroups[layerName]) {{
                    layerGroups[layerName] = new THREE.Group();
                    layerGroups[layerName].name = layerName;
                    scene.add(layerGroups[layerName]);
                }}

                const color = getLayerColor(layerName);
                const material = new THREE.LineBasicMaterial({{ color: color }});

                let mesh = null;

                if (entity.type === 'LINE') {{
                    if (entity.vertices && entity.vertices.length >= 2) {{
                        const points = [
                            new THREE.Vector3(entity.vertices[0].x, entity.vertices[0].y, 0),
                            new THREE.Vector3(entity.vertices[1].x, entity.vertices[1].y, 0)
                        ];
                        const geometry = new THREE.BufferGeometry().setFromPoints(points);
                        mesh = new THREE.Line(geometry, material);
                        updateBounds(entity.vertices);
                    }}
                }}
                else if (entity.type === 'CIRCLE') {{
                    if (entity.center && entity.radius) {{
                        const segments = Math.max(32, Math.round(entity.radius / 5));
                        const geometry = new THREE.CircleGeometry(entity.radius, segments);
                        const edges = new THREE.EdgesGeometry(geometry);
                        mesh = new THREE.LineSegments(edges, material);
                        mesh.position.set(entity.center.x, entity.center.y, 0);
                        updateBounds([
                            {{ x: entity.center.x - entity.radius, y: entity.center.y - entity.radius }},
                            {{ x: entity.center.x + entity.radius, y: entity.center.y + entity.radius }}
                        ]);
                    }}
                }}
                else if (entity.type === 'ARC') {{
                    if (entity.center && entity.radius) {{
                        const startAngle = (entity.startAngle || 0) * Math.PI / 180;
                        const endAngle = (entity.endAngle || 360) * Math.PI / 180;
                        const curve = new THREE.EllipseCurve(
                            entity.center.x, entity.center.y,
                            entity.radius, entity.radius,
                            startAngle, endAngle,
                            false, 0
                        );
                        const points = curve.getPoints(50);
                        const geometry = new THREE.BufferGeometry().setFromPoints(points);
                        mesh = new THREE.Line(geometry, material);
                        updateBounds([
                            {{ x: entity.center.x - entity.radius, y: entity.center.y - entity.radius }},
                            {{ x: entity.center.x + entity.radius, y: entity.center.y + entity.radius }}
                        ]);
                    }}
                }}
                else if (entity.type === 'LWPOLYLINE' || entity.type === 'POLYLINE') {{
                    if (entity.vertices && entity.vertices.length >= 2) {{
                        const points = entity.vertices.map(v => new THREE.Vector3(v.x, v.y, 0));
                        if (entity.shape || entity.closed) {{
                            points.push(points[0].clone());
                        }}
                        const geometry = new THREE.BufferGeometry().setFromPoints(points);
                        mesh = new THREE.Line(geometry, material);
                        updateBounds(entity.vertices);
                    }}
                }}
                else if (entity.type === 'SPLINE') {{
                    if (entity.controlPoints && entity.controlPoints.length >= 2) {{
                        const points = entity.controlPoints.map(v => new THREE.Vector3(v.x, v.y, 0));
                        const curve = new THREE.CatmullRomCurve3(points);
                        const curvePoints = curve.getPoints(50);
                        const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
                        mesh = new THREE.Line(geometry, material);
                        updateBounds(entity.controlPoints);
                    }}
                }}
                else if (entity.type === 'POINT') {{
                    if (entity.position) {{
                        const geometry = new THREE.CircleGeometry(2, 8);
                        mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({{ color: color }}));
                        mesh.position.set(entity.position.x, entity.position.y, 0);
                        updateBounds([entity.position]);
                    }}
                }}
                else if (entity.type === 'TEXT' || entity.type === 'MTEXT') {{
                    // Text entities - draw as placeholder rectangle
                    if (entity.startPoint || entity.position) {{
                        const pos = entity.startPoint || entity.position;
                        const h = entity.textHeight || 10;
                        const w = (entity.text?.length || 5) * h * 0.6;
                        const points = [
                            new THREE.Vector3(pos.x, pos.y, 0),
                            new THREE.Vector3(pos.x + w, pos.y, 0),
                            new THREE.Vector3(pos.x + w, pos.y + h, 0),
                            new THREE.Vector3(pos.x, pos.y + h, 0),
                            new THREE.Vector3(pos.x, pos.y, 0)
                        ];
                        const geometry = new THREE.BufferGeometry().setFromPoints(points);
                        const textMaterial = new THREE.LineDashedMaterial({{
                            color: color,
                            dashSize: 3,
                            gapSize: 2
                        }});
                        mesh = new THREE.Line(geometry, textMaterial);
                        mesh.computeLineDistances();
                        updateBounds([pos, {{ x: pos.x + w, y: pos.y + h }}]);
                    }}
                }}

                if (mesh) {{
                    layerGroups[layerName].add(mesh);
                    entityCount++;
                }}
            }});
        }}

        function updateBounds(vertices) {{
            vertices.forEach(v => {{
                if (v.x !== undefined && v.y !== undefined) {{
                    bounds.minX = Math.min(bounds.minX, v.x);
                    bounds.maxX = Math.max(bounds.maxX, v.x);
                    bounds.minY = Math.min(bounds.minY, v.y);
                    bounds.maxY = Math.max(bounds.maxY, v.y);
                }}
            }});
        }}

        function buildLayerPanel() {{
            const list = document.getElementById('layer-list');
            list.innerHTML = '';

            Object.keys(layerGroups).sort().forEach(layerName => {{
                const color = getLayerColor(layerName);
                const hexColor = '#' + color.toString(16).padStart(6, '0');

                const item = document.createElement('div');
                item.className = 'layer-item';
                item.innerHTML = `
                    <input type="checkbox" checked id="layer-${{layerName}}"
                           onchange="toggleLayer('${{layerName}}', this.checked)">
                    <span class="layer-color" style="background: ${{hexColor}}"></span>
                    <label for="layer-${{layerName}}">${{layerName}}</label>
                `;
                list.appendChild(item);
            }});
        }}

        function toggleLayer(layerName, visible) {{
            if (layerGroups[layerName]) {{
                layerGroups[layerName].visible = visible;
            }}
        }}

        function toggleLayerPanel() {{
            const panel = document.getElementById('layer-panel');
            const btn = document.getElementById('layers-btn');
            panel.classList.toggle('visible');
            btn.classList.toggle('active', panel.classList.contains('visible'));
        }}

        function toggleGrid() {{
            gridVisible = !gridVisible;
            gridHelper.visible = gridVisible;
            document.getElementById('grid-btn').classList.toggle('active', gridVisible);
        }}

        function fitToView() {{
            if (bounds.minX === Infinity) return;

            const container = document.getElementById('canvas-container');
            const aspect = container.clientWidth / container.clientHeight;

            const cx = (bounds.minX + bounds.maxX) / 2;
            const cy = (bounds.minY + bounds.maxY) / 2;
            const width = bounds.maxX - bounds.minX;
            const height = bounds.maxY - bounds.minY;

            let viewWidth, viewHeight;
            if (width / height > aspect) {{
                viewWidth = width * 1.1;
                viewHeight = viewWidth / aspect;
            }} else {{
                viewHeight = height * 1.1;
                viewWidth = viewHeight * aspect;
            }}

            camera.left = cx - viewWidth / 2;
            camera.right = cx + viewWidth / 2;
            camera.top = cy + viewHeight / 2;
            camera.bottom = cy - viewHeight / 2;
            camera.updateProjectionMatrix();
        }}

        function resetView() {{
            const container = document.getElementById('canvas-container');
            const aspect = container.clientWidth / container.clientHeight;
            camera.left = -500 * aspect;
            camera.right = 500 * aspect;
            camera.top = 500;
            camera.bottom = -500;
            camera.updateProjectionMatrix();
        }}

        function setupControls() {{
            const canvas = renderer.domElement;
            let isDragging = false;
            let lastX, lastY;

            canvas.addEventListener('mousedown', e => {{
                isDragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
                canvas.style.cursor = 'grabbing';
            }});

            canvas.addEventListener('mousemove', e => {{
                if (!isDragging) return;
                const dx = e.clientX - lastX;
                const dy = e.clientY - lastY;
                const scale = (camera.right - camera.left) / canvas.clientWidth;
                camera.left -= dx * scale;
                camera.right -= dx * scale;
                camera.top += dy * scale;
                camera.bottom += dy * scale;
                camera.updateProjectionMatrix();
                lastX = e.clientX;
                lastY = e.clientY;
            }});

            canvas.addEventListener('mouseup', () => {{
                isDragging = false;
                canvas.style.cursor = 'grab';
            }});

            canvas.addEventListener('mouseleave', () => {{
                isDragging = false;
                canvas.style.cursor = 'default';
            }});

            canvas.addEventListener('mouseenter', () => {{
                canvas.style.cursor = 'grab';
            }});

            canvas.addEventListener('wheel', e => {{
                e.preventDefault();
                const factor = e.deltaY > 0 ? 1.1 : 0.9;

                // Zoom towards mouse position
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                const worldX = camera.left + (mouseX / canvas.clientWidth) * (camera.right - camera.left);
                const worldY = camera.top - (mouseY / canvas.clientHeight) * (camera.top - camera.bottom);

                const newWidth = (camera.right - camera.left) * factor;
                const newHeight = (camera.top - camera.bottom) * factor;

                const ratioX = (worldX - camera.left) / (camera.right - camera.left);
                const ratioY = (camera.top - worldY) / (camera.top - camera.bottom);

                camera.left = worldX - ratioX * newWidth;
                camera.right = camera.left + newWidth;
                camera.top = worldY + ratioY * newHeight;
                camera.bottom = camera.top - newHeight;

                camera.updateProjectionMatrix();
            }}, {{ passive: false }});

            // Handle resize
            window.addEventListener('resize', () => {{
                const container = document.getElementById('canvas-container');
                renderer.setSize(container.clientWidth, container.clientHeight);
                const aspect = container.clientWidth / container.clientHeight;
                const viewHeight = camera.top - camera.bottom;
                const cx = (camera.left + camera.right) / 2;
                const cy = (camera.top + camera.bottom) / 2;
                camera.left = cx - viewHeight * aspect / 2;
                camera.right = cx + viewHeight * aspect / 2;
                camera.updateProjectionMatrix();
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}

        // Initialize on load
        init();
    </script>
</body>
</html>
'''


def display_dxf_viewer(dxf_bytes: bytes, height: int = 500) -> None:
    """
    Display DXF viewer in Streamlit.

    Args:
        dxf_bytes: Raw DXF file bytes
        height: Viewer height in pixels
    """
    import streamlit.components.v1 as components

    html = generate_viewer_html(dxf_bytes, height)
    components.html(html, height=height + 20, scrolling=False)
