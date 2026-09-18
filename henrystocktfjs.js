const http = require('http');
const { WebSocketServer } = require('ws');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm');

function randomNormal() {
    return tf.randomNormal([]);
}

class HenryStock {
    constructor(name, config) {
        this.name = name;
        const initialPrice = config.initialPrice || 500;
        this.currentPrice = tf.scalar(initialPrice);
        this.initialPrice = tf.scalar(initialPrice);
        this.driftDecay = config.driftDecay !== undefined ? config.driftDecay : 0.70;
        this.baseVol = config.baseVol !== undefined ? config.baseVol : 0.02;
        this.expoSensitivity = config.expoSensitivity !== undefined ? config.expoSensitivity : 1.5;
        this.color = config.color || '#64748b';
        this.returnsHistory = tf.zeros([5]);
        this.marketSensitivity = config.marketSensitivity !== undefined ? config.marketSensitivity : 1.0;
        this.marketImpact = config.marketImpact !== undefined ? config.marketImpact : 0.3;
        this.upwardBias = config.upwardBias !== undefined ? config.upwardBias : 0.02;

        // Catmull-Rom 웨이포인트 설정
        this.waypointInterval = config.waypointInterval || 120; // 몇 틱마다 웨이포인트 생성
        this.meanReversionStrength = config.meanReversionStrength || 0.3; // 평균 회귀 강도 (0~1)
        this.randomWalkScale = config.randomWalkScale || 0.15;  // 랜덤 편차 스케일 (log 단위)

        this._tick = 0;
        // 웨이포인트는 로그 가격으로 관리
        const logInit = Math.log(initialPrice);
        // p-1, p0, p1, p2 4개 확보 (Catmull-Rom은 4점 필요)
        this._waypoints = tf.tensor1d([logInit, logInit, logInit, logInit]);
        this._segmentStart = 0; // 현재 구간의 시작 틱
    }

    // Catmull-Rom 보간 (t: 0~1, p0~p3: 4개 제어점)
    _catmullRom(t, p0, p1, p2, p3) {
        const tTensor = tf.scalar(t);
        const t2 = tf.square(tTensor);
        const t3 = tf.mul(t2, tTensor);
        const term1 = tf.mul(tf.scalar(2), p1);
        const term2 = tf.mul(tf.add(tf.neg(p0), p2), tTensor);
        const term3 = tf.mul(
            tf.add(
                tf.sub(tf.mul(tf.scalar(2), p0), tf.mul(tf.scalar(5), p1)),
                tf.sub(tf.mul(tf.scalar(4), p2), p3)
            ),
            t2
        );
        const term4 = tf.mul(
            tf.add(
                tf.add(tf.neg(p0), tf.mul(tf.scalar(3), p1)),
                tf.add(tf.mul(tf.scalar(-3), p2), p3)
            ),
            t3
        );
        return tf.mul(tf.scalar(0.5), tf.addN([term1, term2, term3, term4]));
    }

    // 다음 웨이포인트 값 생성 (랜덤 + 평균 회귀 혼합)
    _nextWaypoint() {
        return tf.tidy(() => {
            const logCurrent = this._waypoints.slice([3], [1]).squeeze();
            const logMean = tf.log(this.initialPrice);
            const deviation = tf.sub(logCurrent, logMean);
            const reversionForce = tf.mul(
                tf.mul(tf.neg(deviation), tf.scalar(this.meanReversionStrength)),
                tf.add(tf.scalar(1), tf.mul(tf.abs(deviation), tf.scalar(2)))
            );
            const randomStep = tf.add(
                tf.mul(randomNormal(), tf.scalar(this.randomWalkScale)),
                tf.scalar(this.upwardBias || 0.02)
            );
            return tf.add(logCurrent, tf.add(randomStep, reversionForce));
        });
    }

    nextTick(marketEmaTrend) {
        this._tick++;

        // 구간 경계마다 웨이포인트 슬라이드
        if (this._tick > 0 && (this._tick - this._segmentStart) >= this.waypointInterval) {
            this._segmentStart = this._tick;
            const newWp = this._nextWaypoint();
            const nextWaypoints = tf.tidy(() => tf.concat([
                this._waypoints.slice([1], [3]),
                newWp.reshape([1])
            ]));
            this._waypoints.dispose();
            newWp.dispose();
            this._waypoints = nextWaypoints;
        }

        // 현재 구간 내 t (0~1)
        const t = Math.min((this._tick - this._segmentStart) / this.waypointInterval, 1.0);
        const result = tf.tidy(() => {
            const [p0, p1, p2, p3] = tf.unstack(this._waypoints);
            const dcLog = this._catmullRom(t, p0, p1, p2, p3);
            const zPrice = randomNormal();
            const zVol = randomNormal();
            const instVol = tf.mul(tf.scalar(this.baseVol * 0.25), tf.exp(tf.mul(zVol, tf.scalar(0.2))));
            const trend = tf.scalar(marketEmaTrend);
            const sign = marketEmaTrend >= 0 ? 1 : -1;
            const shaped = tf.mul(tf.scalar(sign), tf.pow(tf.abs(trend), tf.scalar(1 / this.expoSensitivity)));
            const asymmetric = tf.mul(shaped, tf.scalar(marketEmaTrend < 0 ? 1.3 : 1.0));
            const acNoise = tf.add(
                tf.mul(tf.mul(asymmetric, tf.scalar(this.marketSensitivity)), tf.scalar(0.008)),
                tf.mul(zPrice, instVol)
            );
            const nextPrice = tf.maximum(tf.scalar(5), tf.mul(tf.exp(dcLog), tf.add(tf.scalar(1), acNoise)));
            const currentReturn = tf.log(tf.div(nextPrice, this.currentPrice));
            const history = tf.concat([this.returnsHistory, currentReturn.reshape([1])]);
            const historyStart = Math.max(0, history.shape[0] - 20);
            const updatedHistory = history.slice([historyStart], [Math.min(20, history.shape[0])]);
            return {
                nextPrice: tf.keep(nextPrice),
                returnsHistory: tf.keep(updatedHistory),
                currentPrice: nextPrice.dataSync()[0],
                marketVolatility: instVol.dataSync()[0],
                baseTrend: tf.mul(tf.sub(p2, p1), tf.scalar(100 / this.waypointInterval)).dataSync()[0]
            };
        });
        this.currentPrice.dispose();
        this.returnsHistory.dispose();
        this.currentPrice = result.nextPrice;
        this.returnsHistory = result.returnsHistory;

        return {
            name: this.name,
            currentPrice: Number(result.currentPrice.toFixed(2)),
            marketVolatility: Number((result.marketVolatility * 100).toFixed(1)),
            baseTrend: Number(result.baseTrend.toFixed(2)),
            color: this.color
        };
    }

    dispose() {
        this.currentPrice.dispose();
        this.initialPrice.dispose();
        this.returnsHistory.dispose();
        this._waypoints.dispose();
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
            const lastReturnIndex = stock.returnsHistory.shape[0] - 1;
            const lastReturnTensor = stock.returnsHistory.slice([lastReturnIndex], [1]);
            const lastReturn = lastReturnTensor.dataSync()[0] || 0;
            lastReturnTensor.dispose();
            weightedMarketReturn += lastReturn * stock.marketImpact;
            totalImpact += stock.marketImpact;
        });
        
        const currentMarketReturn = totalImpact > 0 ? (weightedMarketReturn / totalImpact) : 0;
        marketEmaTrend = (currentMarketReturn * emaAlpha) + (marketEmaTrend * (1 - emaAlpha));
        const tickData = stocks.map(stock => stock.nextTick(marketEmaTrend));

        const timestampMs = Date.now();
        const timestamp = Math.floor(timestampMs / 1000);

        const payload = {
            event: "stock_update",
            time: timestamp,
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
        stocks.forEach(stock => stock.dispose());
    });
});

const PORT = 3000;

async function startServer() {
    await tf.setBackend('cpu');
    await tf.ready();

    server.listen(PORT, () => {
        console.log(`====================================================`);
        console.log(` 🌐 헨리 가상주식 서버 실행 중... `);
        console.log(` 에이전트 접속 주소: ws://localhost:${PORT}`);
        console.log(`====================================================`);
    });
}

startServer().catch(error => {
    console.error('TensorFlow.js 초기화 실패:', error);
    process.exitCode = 1;
});