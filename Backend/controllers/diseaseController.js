const fs = require("fs").promises;
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Analyze Image Function
async function analyzeImage(file, prompt) {
  const imageData = await fs.readFile(file.path, { encoding: "base64" });

  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  const result = await model.generateContent([
    prompt,
    {
      inlineData: {
        mimeType: file.mimetype,
        data: imageData,
      },
    },
  ]);

  await fs.unlink(file.path); // Clean up uploaded file
  return result.response.text();
}

// Controller Function
async function handleDiseaseDetection(req, res) {
  try {
    const { language } = req.body;

    const marathiDiseasePrompt = `तुम्ही शेतीत तज्ज्ञ आहात जिथे तुम्हाला प्रतिमेतील पानांचे किंवा वनस्पतींचे घटक पहावे लागतील आणि एकूण सांगावे लागतील की हा रोग कोणता आहे, हे का होते आणि प्रतिबंधात्मक उपाय आणि उपाय, तसेच वापरण्यासाठी आवश्यक असलेल्या खतांची माहिती खाली दिलेल्या स्वरूपात द्यावी लागेल.`;

    const englishDiseasePrompt = `You are an expert in Agriculture where you need to see the leaf or plants items from the image and tell the total What disease is this, Why This is Happened and Preventive measure and remedies, also provide the details of Fertilizers Need to use is below format`;

    const prompt =
      language === "mr" ? marathiDiseasePrompt : englishDiseasePrompt;

    const result = await analyzeImage(req.file, prompt);
    res.json({ result });
  } catch (err) {
    console.error("Error analyzing image:", err);
    res.status(500).json({ error: "Failed to analyze image." });
  }
}

// Export
module.exports = { handleDiseaseDetection };
