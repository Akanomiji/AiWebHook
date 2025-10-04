const express = require('express');
const line = require('@line/bot-sdk');
const tf = require('@tensorflow/tfjs-node');

const config = {
    channelAccessToken: process.env.CHANNEL_ACCESS_TOKEN,
    channelSecret: process.env.CHANNEL_SECRET,
};
const modelUrl = 'https://teachablemachine.withgoogle.com/models/FncwMoWqm/model.json'; // <-- อย่าลืมใส่ URL ของโมเดลคุณตรงนี้นะครับ
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

        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(127.5))
            .sub(tf.scalar(1))
            .expandDims();

        const predictionResult = await model.predict(imageTensor).data();

        let bestPrediction = { className: 'ไม่รู้จัก', probability: 0 };
        for (let i = 0; i < predictionResult.length; i++) {
            if (predictionResult[i] > bestPrediction.probability) {
                bestPrediction.probability = predictionResult[i];
                bestPrediction.className = classNames[i];
            }
        }

        const confidence = Math.round(bestPrediction.probability * 100);

        const flexMessage = {
            "type": "flex",
            "altText": `ผลการทำนาย: ${bestPrediction.className} (ความแม่นยำ ${confidence}%)`,
            "contents": {
                "type": "bubble",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "image",
                                    "url": "https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg?semt=ais_hybrid&w=740&q=80", 
                                    "size": "xxs",
                                    "flex": 1,
                                    "gravity": "center"
                                },
                                {
                                    "type": "text",
                                    "text": "ผลการวิเคราะห์รูปภาพ",
                                    "weight": "bold",
                                    "size": "md",
                                    "color": "#1A1A1A",
                                    "flex": 4,
                                    "margin": "md",
                                    "gravity": "center"
                                }
                            ]
                        }
                    ],
                    "paddingAll": "12px"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "md",
                    "contents": [
                        {
                            "type": "separator"
                        },
                        {
                            "type": "box",
                            "layout": "vertical",
                            "margin": "lg",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "box",
                                    "layout": "horizontal",
                                    "contents": [
                                        {
                                            "type": "text",
                                            "text": "โรคที่พบ:",
                                            "size": "sm",
                                            "color": "#555555",
                                            "flex": 0
                                        },
                                        {
                                            "type": "text",
                                            "text": bestPrediction.className,
                                            "size": "sm",
                                            "color": "#111111",
                                            "align": "end",
                                            "weight": "bold"
                                        }
                                    ]
                                },
                                {
                                    "type": "box",
                                    "layout": "horizontal",
                                    "contents": [
                                        {
                                            "type": "text",
                                            "text": "ความแม่นยำ:",
                                            "size": "sm",
                                            "color": "#555555",
                                            "flex": 0
                                        },
                                        {
                                            "type": "text",
                                            "text": `${confidence}%`,
                                            "size": "sm",
                                            "color": "#111111",
                                            "align": "end",
                                            "weight": "bold"
                                        }
                                    ]
                                }
                            ]
                        },
                        {
                            "type": "separator",
                            "margin": "lg"
                        }
                    ]
                },
                "styles": {
                    "header": {
                        "backgroundColor": "#F0F0F0"
                    }
                }
            }
        };

        // ส่ง Flex Message กลับไปหาผู้ใช้
        return client.replyMessage(event.replyToken, flexMessage);

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