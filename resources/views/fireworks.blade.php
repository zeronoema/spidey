<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fireworks</title>
    <style>
        html, body {
            margin: 0;
            height: 100%;
            overflow: hidden;
            background: #000;
        }
        canvas {
            display: block;
            width: 100vw;
            height: 100vh;
            cursor: pointer;
        }
        .hint {
            position: fixed;
            bottom: 16px;
            width: 100%;
            text-align: center;
            font-family: system-ui, sans-serif;
            font-size: 13px;
            color: rgba(255, 255, 255, 0.35);
            pointer-events: none;
            user-select: none;
        }
    </style>
</head>
<body>
    <canvas id="sky"></canvas>
    <div class="hint">Click anywhere to launch again</div>

    <script>
        (() => {
            const canvas = document.getElementById('sky');
            const ctx = canvas.getContext('2d');
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            let W, H;

            function resize() {
                W = window.innerWidth;
                H = window.innerHeight;
                canvas.width = W * dpr;
                canvas.height = H * dpr;
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, W, H);
            }
            window.addEventListener('resize', resize);
            resize();

            const GRAVITY = 0.045;
            const rand = (min, max) => Math.random() * (max - min) + min;

            let rocket = null;
            const sparks = [];
            let flash = 0;
            let relaunchPending = false;

            // Rocket: rises from the ground, slows down under gravity,
            // and bursts at the apex, leaving a flickering smoke/spark trail.
            function launch(targetX, targetY) {
                const startX = targetX ?? W / 2 + rand(-W * 0.1, W * 0.1);
                const peakY = targetY ?? rand(H * 0.18, H * 0.32);
                const height = H - peakY;
                // v^2 = 2gh -> initial speed needed to reach the peak
                const vy = -Math.sqrt(2 * GRAVITY * height);
                rocket = {
                    x: startX,
                    y: H,
                    vx: rand(-0.3, 0.3),
                    vy,
                    hue: rand(0, 360),
                };
            }

            function explode(x, y, baseHue) {
                flash = 1;
                const count = 260;
                // A peony shell: particles spread evenly on a sphere,
                // projected to 2D so the burst looks round and full.
                for (let i = 0; i < count; i++) {
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.acos(rand(-1, 1));
                    const speed = rand(3.6, 4.2);
                    const vx = Math.sin(phi) * Math.cos(theta) * speed;
                    const vy = Math.sin(phi) * Math.sin(theta) * speed;
                    sparks.push({
                        x, y,
                        px: x, py: y,
                        vx, vy,
                        hue: baseHue + rand(-12, 12),
                        life: 1,
                        decay: rand(0.006, 0.010),
                        size: rand(1.2, 2.2),
                        drag: 0.965,
                        crackle: Math.random() < 0.35,
                        trail: true,
                    });
                }
                // Inner core of white-gold glitter
                for (let i = 0; i < 70; i++) {
                    const a = Math.random() * Math.PI * 2;
                    const s = rand(0.5, 1.8);
                    sparks.push({
                        x, y,
                        px: x, py: y,
                        vx: Math.cos(a) * s,
                        vy: Math.sin(a) * s,
                        hue: 45,
                        sat: 30,
                        life: 1,
                        decay: rand(0.015, 0.025),
                        size: rand(0.8, 1.4),
                        drag: 0.95,
                        crackle: true,
                        trail: false,
                    });
                }
            }

            function emberTrail(x, y) {
                sparks.push({
                    x: x + rand(-1, 1),
                    y: y + rand(0, 3),
                    px: x, py: y,
                    vx: rand(-0.25, 0.25),
                    vy: rand(0.2, 0.8),
                    hue: 35,
                    sat: 90,
                    life: rand(0.5, 0.8),
                    decay: rand(0.02, 0.04),
                    size: rand(0.6, 1.2),
                    drag: 0.96,
                    crackle: true,
                    trail: false,
                });
            }

            function frame() {
                // Translucent overlay instead of a full clear leaves soft motion trails
                ctx.globalCompositeOperation = 'source-over';
                ctx.fillStyle = 'rgba(0, 0, 0, 0.18)';
                ctx.fillRect(0, 0, W, H);

                // Sky lights up briefly when the shell bursts
                if (flash > 0.01) {
                    ctx.fillStyle = `rgba(255, 240, 220, ${flash * 0.12})`;
                    ctx.fillRect(0, 0, W, H);
                    flash *= 0.85;
                }

                ctx.globalCompositeOperation = 'lighter';

                if (rocket) {
                    const r = rocket;
                    const px = r.x, py = r.y;
                    r.vy += GRAVITY;
                    r.x += r.vx;
                    r.y += r.vy;

                    ctx.strokeStyle = 'rgba(255, 210, 150, 0.9)';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(r.x, r.y);
                    ctx.stroke();
                    emberTrail(r.x, r.y);
                    emberTrail(r.x, r.y);

                    if (r.vy >= -0.4) {
                        explode(r.x, r.y, r.hue);
                        rocket = null;
                    }
                }

                for (let i = sparks.length - 1; i >= 0; i--) {
                    const p = sparks[i];
                    p.px = p.x;
                    p.py = p.y;
                    p.vx *= p.drag;
                    p.vy = p.vy * p.drag + GRAVITY * 0.6;
                    p.x += p.vx;
                    p.y += p.vy;
                    p.life -= p.decay;

                    if (p.life <= 0) {
                        sparks.splice(i, 1);
                        continue;
                    }

                    // Crackling sparks flicker on and off as they die out
                    if (p.crackle && p.life < 0.6 && Math.random() < 0.4) continue;

                    const sat = p.sat ?? 100;
                    // Hot sparks start near-white and cool into their colour
                    const light = 50 + p.life * 35;
                    const alpha = Math.min(1, p.life * 1.4);
                    const color = `hsla(${p.hue}, ${sat}%, ${light}%, ${alpha})`;

                    if (p.trail) {
                        ctx.strokeStyle = color;
                        ctx.lineWidth = p.size;
                        ctx.lineCap = 'round';
                        ctx.beginPath();
                        ctx.moveTo(p.px, p.py);
                        ctx.lineTo(p.x, p.y);
                        ctx.stroke();
                    }

                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size * (0.6 + p.life * 0.6), 0, Math.PI * 2);
                    ctx.fill();
                }

                // Auto relaunch a moment after the previous firework has faded
                if (!rocket && sparks.length === 0 && !relaunchPending) {
                    relaunchPending = true;
                    setTimeout(() => {
                        relaunchPending = false;
                        if (!rocket && sparks.length === 0) launch();
                    }, 900);
                }

                requestAnimationFrame(frame);
            }

            canvas.addEventListener('click', (e) => {
                if (rocket) return;
                launch(e.clientX, e.clientY);
            });

            launch();
            requestAnimationFrame(frame);
        })();
    </script>
</body>
</html>
