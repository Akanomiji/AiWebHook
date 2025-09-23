const express = require('express');
const line = require('@line/bot-sdk');
const tf = require('@tensorflow/tfjs-node');

// ===============================================================
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
//                  >>> ส่วนที่ต้องแก้ไข <<<
// ===============================================================
const config = {
    channelAccessToken: process.env.CHANNEL_ACCESS_TOKEN || 'ViemuhahPnVU7sx6PHn7o38oIMAN53NCCTNyn/MOSw2kEBCrDjblMu3dAXJGOiiXjmGZKEzGSh8+Qu34m0zFdgDcBE/N/DRbhhq3vGN55ABI4KI6WCaEH1mYbaRX0+ExJREvPmWeM1dkN39TX5FhlQdB04t89/1O/w1cDnyilFU=',
    channelSecret: process.env.CHANNEL_SECRET || '64f431e7e5bdf81b3d629aaa8eed6a6d',
};
const modelUrl = 'https://teachablemachine.withgoogle.com/models/1i_v_rE_a/model.json';
const classNames = ['healthy', 'mosaic virus', 'Yellow Leaf Curl Virus', 'Target Spot', 'Spider mites', 'Septoria leaf spot','Leaf Mold','Late blight','Early blight','Bacterial spot']; // !!! แก้ไขให้ตรงกับโมเดลของคุณ !!!
// ===============================================================
// ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
// ===============================================================

const app = express();
const client = new line.Client(config);
let model;

app.post('/webhook', line.middleware(config), (req, res) => {
    Promise.all(req.body.events.map(handleEvent))
        .then((result) => res.json(result))
        .catch((err) => {
            console.error(err);
            res.status(500).end();
        });
});

async function handleEvent(event) {
    if (event.type !== 'message' || event.message.type !== 'image') {
        return Promise.resolve(null);
    }
    try {
        const imageBuffer = await getImageBufferFromLine(event.message.id);
        const imageTensor = tf.node.decodeImage(imageBuffer, 3).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
        const predictionResult = await model.predict(imageTensor).data();

        let bestPrediction = { className: 'ไม่รู้จัก', probability: 0 };
        for (let i = 0; i < predictionResult.length; i++) {
            if (predictionResult[i] > bestPrediction.probability) {
                bestPrediction.probability = predictionResult[i];
                bestPrediction.className = classNames[i];
            }
        }

        const confidence = Math.round(bestPrediction.probability * 100);
        const replyText = `ฉันคิดว่ารูปนี้คือ "${bestPrediction.className}" นะ! (ความแม่นยำ ${confidence}%)`;

        return client.replyMessage(event.replyToken, { type: 'text', text: replyText });
    } catch (error) {
        console.error('เกิดข้อผิดพลาด:', error);
        return client.replyMessage(event.replyToken, { type: 'text', text: 'ขออภัยค่ะ เกิดข้อผิดพลาดบางอย่าง' });
    }
}

function getImageBufferFromLine(messageId) {
    return new Promise((resolve, reject) => {
        client.getMessageContent(messageId)
            .then((stream) => {
                const chunks = [];
                stream.on('data', (chunk) => { chunks.push(chunk); });
                stream.on('error', (err) => { reject(err); });
                stream.on('end', () => { resolve(Buffer.concat(chunks)); });
            });
    });
}

async function startServer() {
    try {
        console.log('กำลังโหลดโมเดล...');
        model = await tf.loadLayersModel(modelUrl);
        console.log('โหลดโมเดลสำเร็จ!');
        const port = process.env.PORT || 3000;
        app.listen(port, () => {
            console.log(`บอทพร้อมทำงานที่ port ${port}`);
        });
    } catch (error) {
        console.error('ไม่สามารถโหลดโมเดลได้:', error);
    }
}

startServer();