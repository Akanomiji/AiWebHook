const express = require('express');
const line = require('@line/bot-sdk');
const tf = require('@tensorflow/tfjs-node');

const config = {
    channelAccessToken: process.env.CHANNEL_ACCESS_TOKEN,
    channelSecret: process.env.CHANNEL_SECRET,
};
const modelUrl = 'https://teachablemachine.withgoogle.com/models/FncwMoWqm/model.json';
const classNames = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'];

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
        console.error(error);
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
        console.log('Loading model...');
        model = await tf.loadLayersModel(modelUrl);
        console.log('Model loaded!');
        const port = process.env.PORT || 3000;
        app.listen(port, () => {
            console.log(`Bot is ready on port ${port}`);
        });
    } catch (error) {
        console.error('Failed to load model:', error);
    }
}

startServer();