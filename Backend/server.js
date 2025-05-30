  
    require("dotenv").config();
    const express = require("express");
    const multer = require("multer");
    const cors = require("cors");
    const bodyParser = require("body-parser");
    const path = require("path");
    

    const { handleDiseaseDetection } = require("./controllers/diseaseController");
    const { handleCropMonitoring } = require("./controllers/monitoringController");

    const app = express();
    const port = process.env.PORT || 5000;

    // Middlewares
    app.use(cors());
    app.use(bodyParser.json())
    app.use(express.json({ limit: "10mb" }));

    // Multer Setup
    const upload = multer({ dest: "uploads/" });

    // Routes
    app.post("/disease", upload.single("image"), handleDiseaseDetection);
    app.post("/cropmonitoring", handleCropMonitoring);

    // Start server
    app.listen(port, () => console.log(`Server running on port ${port}`));