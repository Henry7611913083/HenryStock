const { readFileSync } = require('fs');
const http = require('http');
const { WebSocketServer } = require('ws');
const fs = require('fs');

function randomNormal() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); 
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

class HenryStock {
    constructor(name, config) {
        this.name = name;
        this.currentPrice = config.initialPrice || 500;
        this.initialPrice = config.initialPrice || 500;
        this.driftDecay = config.driftDecay !== undefined ? config.driftDecay : 0.70;
        this.baseVol = config.baseVol !== undefined ? config.baseVol : 0.02;
        this.expoSensitivity = config.expoSensitivity !== undefined ? config.expoSensitivity : 1.5;
        this.color = config.color || '#64748b';
        this.returnsHistory = [0.0, 0.0, 0.0, 0.0, 0.0];
        this.marketSensitivity = config.marketSensitivity !== undefined ? config.marketSensitivity : 1.0;
        this.marketImpact = config.marketImpact !== undefined ? config.marketImpact : 0.3;
        this.upwardBias = config.upwardBias !== undefined ? config.upwardBias : 0.02;

        // Catmull-Rom 웨이포인트 설정
        this.waypointInterval = config.waypointInterval || 120; // 몇 틱마다 웨이포인트 생성
        this.meanReversionStrength = config.meanReversionStrength || 0.3; // 평균 회귀 강도 (0~1)
        this.randomWalkScale = config.randomWalkScale || 0.15;  // 랜덤 편차 스케일 (log 단위)

        this._tick = 0;
        // 웨이포인트는 로그 가격으로 관리
        const logInit = Math.log(this.initialPrice);
        // p-1, p0, p1, p2 4개 확보 (Catmull-Rom은 4점 필요)
        this._waypoints = [logInit, logInit, logInit, logInit];
        this._segmentStart = 0; // 현재 구간의 시작 틱
    }

    // Catmull-Rom 보간 (t: 0~1, p0~p3: 4개 제어점)
    _catmullRom(t, p0, p1, p2, p3) {
        const t2 = t * t;
        const t3 = t2 * t;
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        );
    }

    // 다음 웨이포인트 값 생성 (랜덤 + 평균 회귀 혼합)
    _nextWaypoint() {
        const logCurrent = this._waypoints[this._waypoints.length - 1];
        const logMean = Math.log(this.initialPrice);
        const deviation = logCurrent - logMean; // 현재 평균 대비 얼마나 벗어났는지
    
        // 편차가 클수록 회귀력이 강해지는 비선형 회귀
        const reversionForce = -deviation * this.meanReversionStrength
            * (1 + Math.abs(deviation) * 2.0); // 멀어질수록 더 강하게 당김
    
        // 랜덤 편차: 상승/하락 비대칭 (상승 편향 추가)
        const rawRandom = randomNormal() * this.randomWalkScale;
        const upwardBias = this.upwardBias || 0.02; // stock.json에서 조정
        const randomStep = rawRandom + upwardBias;
    
        return logCurrent + randomStep + reversionForce;
    }

    nextTick(marketEmaTrend) {
        this._tick++;

        // 구간 경계마다 웨이포인트 슬라이드
        if (this._tick > 0 && (this._tick - this._segmentStart) >= this.waypointInterval) {
            this._segmentStart = this._tick;
            const newWp = this._nextWaypoint();
            this._waypoints.shift();
            this._waypoints.push(newWp);
        }

        // 현재 구간 내 t (0~1)
        const t = Math.min((this._tick - this._segmentStart) / this.waypointInterval, 1.0);
        const [p0, p1, p2, p3] = this._waypoints;

        // DC: Catmull-Rom 보간된 로그 가격
        const dcLog = this._catmullRom(t, p0, p1, p2, p3);

        // AC: 시장 감응 + 노이즈 (작고 빠르게)
        const zPrice = randomNormal();
        const zVol = randomNormal();
        const instVol = this.baseVol * 0.25 * Math.exp(zVol * 0.2);

        const sign = marketEmaTrend >= 0 ? 1 : -1;
        const shaped = sign * Math.pow(Math.abs(marketEmaTrend), 1 / this.expoSensitivity);
        const asymmetric = shaped * (marketEmaTrend < 0 ? 1.3 : 1.0);
        const acNoise = (asymmetric * this.marketSensitivity * 0.008) + (zPrice * instVol);

        // 최종 가격: DC 기준에 AC를 곱셈으로 얹기
        let nextPrice = Math.exp(dcLog) * (1 + acNoise);

        // 하한선 방어
        if (nextPrice < 5) nextPrice = 5;

        const currentReturn = Math.log(nextPrice / this.currentPrice);
        this.returnsHistory.push(currentReturn);
        if (this.returnsHistory.length > 20) this.returnsHistory.shift();
        this.currentPrice = nextPrice;

        return {
            name: this.name,
            currentPrice: Number(this.currentPrice.toFixed(2)),
            marketVolatility: Number((instVol * 100).toFixed(1)),
            baseTrend: Number(((p2 - p1) / this.waypointInterval * 100).toFixed(2)),
            color: this.color
        };
    }
}

// 주식 개별 파라미터 설정
const stockConfigs = require('./stock.json');

const server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('시장 EMA 필터가 적용된 웹소켓 주식 서버 실행 중...');
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
    console.log("⚡ 에이전트가 연결되었습니다. (시장 연동 모드)");

    const stocks = stockConfigs.map(config => new HenryStock(config.name, config));
    let intervalId = null;
    let currentInterval = 1000; 
    let isPaused = false; 

    let marketEmaTrend = 0.0;
    const emaAlpha = 0.2;

    const sendTick = () => {
        if (isPaused) return;

        let totalImpact = 0;
        let weightedMarketReturn = 0;

        stocks.forEach(stock => {
            const lastReturn = stock.returnsHistory[stock.returnsHistory.length - 1] || 0;
            weightedMarketReturn += lastReturn * stock.marketImpact;
            totalImpact += stock.marketImpact;
        });
        
        const currentMarketReturn = totalImpact > 0 ? (weightedMarketReturn / totalImpact) : 0;
        marketEmaTrend = (currentMarketReturn * emaAlpha) + (marketEmaTrend * (1 - emaAlpha));
        const tickData = stocks.map(stock => stock.nextTick(marketEmaTrend));

        const timestampMs = Date.now();

        const payload = {
            event: "stock_update",
            time: timestampMs,
            timestampMs,
            stocks: tickData,
            marketTrend: Number((marketEmaTrend * 100).toFixed(3)),
            interval: currentInterval
        };

        if (ws.readyState === ws.OPEN) {
            ws.send(JSON.stringify(payload));
        }
    };

    const startSimulationTimer = (ms) => {
        if (intervalId) clearInterval(intervalId);
        currentInterval = ms;
        intervalId = setInterval(sendTick, ms);
    };

    startSimulationTimer(currentInterval);

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            if (data.event === "set_interval" && typeof data.interval === "number") {
                const newInterval = Math.max(100, data.interval);
                startSimulationTimer(newInterval);
            }
            if (data.event === "toggle_pause") {
                isPaused = !isPaused;
                console.log(`⏸️ 시뮬레이션 상태 변경: ${isPaused ? '일시정지' : '재개'}`);
                ws.send(JSON.stringify({ event: "pause_status", isPaused: isPaused }));
            }
        } catch (err) {
            console.error("클라이언트 메시지 파싱 에러:", err);
        }
    });

    ws.on('close', () => {
        console.log("❌ 에이전트가 연결을 끊었습니다.");
        clearInterval(intervalId);
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`====================================================`);
    console.log(` 🌐 헨리 가상주식 서버 실행 중... `);
    console.log(` 에이전트 접속 주소: ws://localhost:${PORT}`);
    console.log(`====================================================`);
});