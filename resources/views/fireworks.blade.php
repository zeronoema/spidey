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
        #flash {
            position: fixed;
            inset: 0;
            background: rgb(255, 235, 210);
            opacity: 0;
            pointer-events: none;
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
    <div id="flash"></div>
    <div class="hint">Click anywhere to launch more</div>

    <script>
        (() => {
            const canvas = document.getElementById('sky');
            const ctx = canvas.getContext('2d');
            const flashLayer = document.getElementById('flash');
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
            const MAX_SPARKS = 6000;
            const SHELL_TYPES = ['peony', 'peony', 'chrysanthemum', 'willow', 'ring', 'crossette'];
            const rand = (min, max) => Math.random() * (max - min) + min;
            const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

            const rockets = [];
            const sparks = [];
            let flash = 0;

            // Rocket: rises from the ground, slows down under gravity,
            // and bursts at the apex, leaving a flickering spark trail.
            function launch(targetX, targetY, type) {
                const startX = targetX ?? rand(W * 0.1, W * 0.9);
                const peakY = targetY ?? rand(H * 0.1, H * 0.45);
                const height = H - peakY;
                // v^2 = 2gh -> initial speed needed to reach the peak
                const vy = -Math.sqrt(2 * GRAVITY * height);
                rockets.push({
                    x: type === 'heart' ? startX : startX + rand(-40, 40),
                    y: H,
                    vx: type === 'heart' ? 0 : rand(-0.4, 0.4),
                    vy,
                    hue: rand(0, 360),
                    type: type ?? pick(SHELL_TYPES),
                });
            }

            function addSpark(s) {
                if (sparks.length < MAX_SPARKS) sparks.push(s);
            }

            function explode(r) {
                const { x, y, hue: baseHue, type } = r;
                flash = Math.min(1, flash + 0.5);

                if (type === 'heart') {
                    // Heart outline from the classic parametric curve, plus a softer filled
                    // inner heart. All stars share the same drag, so the shape holds while it expands.
                    const scale = 0.33;
                    const layers = [
                        { f: 1,    n: 160, hue: 345, size: 2.0 },
                        { f: 0.7,  n: 90,  hue: 330, size: 1.6 },
                        { f: 0.4,  n: 50,  hue: 350, size: 1.3 },
                    ];
                    for (const l of layers) {
                        for (let i = 0; i < l.n; i++) {
                            const t = (i / l.n) * Math.PI * 2;
                            const hx = 16 * Math.sin(t) ** 3;
                            const hy = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));
                            const jitter = rand(0.97, 1.03);
                            addSpark(makeSpark(x, y, hx * scale * l.f * jitter, hy * scale * l.f * jitter, {
                                hue: l.hue + rand(-6, 6),
                                sat: 100,
                                decay: rand(0.006, 0.008),
                                size: l.size,
                                drag: 0.955,
                                gravity: 0.25,
                                crackle: l.f < 1,
                            }));
                        }
                    }
                    flash = 1;
                    return;
                }

                if (type === 'ring') {
                    // Flat ring tilted at a random angle
                    const tilt = rand(0.3, 1);
                    const rot = rand(0, Math.PI);
                    for (let i = 0; i < 90; i++) {
                        const a = (i / 90) * Math.PI * 2;
                        const rx = Math.cos(a) * 3.8;
                        const ry = Math.sin(a) * 3.8 * tilt;
                        addSpark(makeSpark(x, y,
                            rx * Math.cos(rot) - ry * Math.sin(rot),
                            rx * Math.sin(rot) + ry * Math.cos(rot),
                            { hue: baseHue + rand(-8, 8), decay: rand(0.008, 0.012) }));
                    }
                    return;
                }

                const opts = {
                    peony:         { count: 220, speed: [3.4, 4.0], decay: [0.007, 0.011], drag: 0.965, hue: baseHue, sat: 100 },
                    chrysanthemum: { count: 260, speed: [3.8, 4.4], decay: [0.005, 0.008], drag: 0.97,  hue: baseHue, sat: 90 },
                    willow:        { count: 200, speed: [2.8, 3.4], decay: [0.003, 0.005], drag: 0.975, hue: 38,      sat: 85 },
                    crossette:     { count: 24,  speed: [3.0, 3.4], decay: [0.018, 0.022], drag: 0.97,  hue: baseHue, sat: 100 },
                }[type];

                // Particles spread evenly on a sphere, projected to 2D so the burst looks round and full.
                for (let i = 0; i < opts.count; i++) {
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.acos(rand(-1, 1));
                    const speed = rand(...opts.speed);
                    addSpark(makeSpark(x, y,
                        Math.sin(phi) * Math.cos(theta) * speed,
                        Math.sin(phi) * Math.sin(theta) * speed,
                        {
                            hue: opts.hue + rand(-12, 12),
                            sat: opts.sat,
                            decay: rand(...opts.decay),
                            drag: opts.drag,
                            crackle: type === 'willow' || Math.random() < 0.3,
                            size: type === 'willow' ? rand(0.9, 1.5) : rand(1.2, 2.2),
                            gravity: type === 'willow' ? 1.1 : 0.6,
                            splits: type === 'crossette',
                        }));
                }

                // Inner core of white-gold glitter
                if (type !== 'crossette') {
                    for (let i = 0; i < 50; i++) {
                        const a = Math.random() * Math.PI * 2;
                        const s = rand(0.5, 1.8);
                        addSpark(makeSpark(x, y, Math.cos(a) * s, Math.sin(a) * s, {
                            hue: 45, sat: 30,
                            decay: rand(0.015, 0.025),
                            size: rand(0.8, 1.4),
                            drag: 0.95,
                            crackle: true,
                            trail: false,
                        }));
                    }
                }
            }

            function makeSpark(x, y, vx, vy, o) {
                return {
                    x, y, px: x, py: y, vx, vy,
                    hue: o.hue ?? 40,
                    sat: o.sat ?? 100,
                    life: o.life ?? 1,
                    decay: o.decay ?? 0.01,
                    size: o.size ?? rand(1.2, 2.2),
                    drag: o.drag ?? 0.965,
                    gravity: o.gravity ?? 0.6,
                    crackle: o.crackle ?? false,
                    trail: o.trail ?? true,
                    splits: o.splits ?? false,
                };
            }

            function emberTrail(x, y) {
                addSpark(makeSpark(x + rand(-1, 1), y + rand(0, 3), rand(-0.25, 0.25), rand(0.2, 0.8), {
                    hue: 35, sat: 90,
                    life: rand(0.5, 0.8),
                    decay: rand(0.02, 0.04),
                    size: rand(0.6, 1.2),
                    drag: 0.96,
                    crackle: true,
                    trail: false,
                }));
            }

            function frame() {
                // Translucent overlay instead of a full clear leaves soft motion trails
                ctx.globalCompositeOperation = 'source-over';
                ctx.fillStyle = 'rgba(0, 0, 0, 0.18)';
                ctx.fillRect(0, 0, W, H);

                // Sky lights up briefly when shells burst (separate layer so it never builds up into haze)
                flashLayer.style.opacity = flash * 0.07;
                flash *= 0.85;

                ctx.globalCompositeOperation = 'lighter';

                for (let i = rockets.length - 1; i >= 0; i--) {
                    const r = rockets[i];
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

                    if (r.vy >= -0.4) {
                        explode(r);
                        rockets.splice(i, 1);
                    }
                }

                for (let i = sparks.length - 1; i >= 0; i--) {
                    const p = sparks[i];
                    p.px = p.x;
                    p.py = p.y;
                    p.vx *= p.drag;
                    p.vy = p.vy * p.drag + GRAVITY * p.gravity;
                    p.x += p.vx;
                    p.y += p.vy;
                    p.life -= p.decay;

                    if (p.life <= 0) {
                        // Crossette stars split into four smaller stars when they burn out
                        if (p.splits) {
                            for (let k = 0; k < 4; k++) {
                                const a = k * Math.PI / 2 + rand(-0.2, 0.2);
                                sparks.push(makeSpark(p.x, p.y, Math.cos(a) * 2, Math.sin(a) * 2, {
                                    hue: p.hue, decay: rand(0.015, 0.022), size: 1.3, crackle: true,
                                }));
                            }
                        }
                        sparks.splice(i, 1);
                        continue;
                    }

                    // Crackling sparks flicker on and off as they die out
                    if (p.crackle && p.life < 0.6 && Math.random() < 0.4) continue;

                    // Hot sparks start near-white and cool into their colour
                    const light = 45 + p.life * 25;
                    const alpha = Math.min(1, p.life * 1.4);
                    const color = `hsla(${p.hue}, ${p.sat}%, ${light}%, ${alpha})`;

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

                requestAnimationFrame(frame);
            }

            // Steady stream of single launches, with an occasional salvo of several at once
            function scheduleLaunch() {
                if (Math.random() < 0.15) {
                    const n = Math.floor(rand(4, 8));
                    for (let i = 0; i < n; i++) setTimeout(launch, i * rand(60, 140));
                } else {
                    launch();
                }
                setTimeout(scheduleLaunch, rand(150, 500));
            }

            canvas.addEventListener('click', (e) => {
                for (let i = 0; i < 3; i++) {
                    setTimeout(() => launch(e.clientX, e.clientY + rand(-40, 40)), i * 120);
                }
            });

            // The love firework: a big heart in the middle of the sky every few seconds
            function launchHeart() {
                launch(W / 2, H * 0.35, 'heart');
                setTimeout(launchHeart, 6000);
            }

            for (let i = 0; i < 5; i++) setTimeout(launch, i * 150);
            setTimeout(launchHeart, 1200);
            scheduleLaunch();
            requestAnimationFrame(frame);
        })();
    </script>
</body>
</html>
