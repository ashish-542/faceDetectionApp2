const express = require("express");
const puppeteer = require('puppeteer');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const serveStatic = require("serve-static");
const { join } = require("path");
const app = express();
const PORT=3000;
// Serve static files from the "public" directory
app.use(express.static("public"));

// Serve the models directory
app.use("/models", serveStatic(join(__dirname, "public", "models")));

function loadMtcnnModel() {
    // Load models asynchronously when the server starts
    console.log("Loading models from file system");
    const modelsPath = join(__dirname, "public", "models");
    Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromDisk(`${modelsPath}/ssd_mobilenetv1_model-weights_manifest.json`),
        faceapi.nets.faceLandmark68Net.loadFromDisk(`${modelsPath}/face_landmark_68_model-weights_manifest.json`),
        faceapi.nets.faceRecognitionNet.loadFromDisk(`${modelsPath}/face_recognition_model-weights_manifest.json`),
    ]).then(startServer);
}

function startServer() {
    app.listen(PORT, () => {
        console.log(`Server is running on port ${PORT}`);
    });
}

// Function to process face detection on a local image
async function detectFacesInImage(imagePath) {
    const image = await canvas.loadImage(imagePath);
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
    return detections;
}

app.get("/", async (req, res) => {
    try {
        const imagePath = __dirname + "/public/images/photo.jpeg";
        const detections = await detectFacesInImage(imagePath);
        res.json(detections);
    } catch (error) {
        console.error("Error processing image:", error);
        res.status(500).send("Error processing image");
    }
});

loadMtcnnModel();
