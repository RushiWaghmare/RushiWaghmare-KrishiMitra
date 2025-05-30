const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Analyze Text Function
// async function analyzeText(inputText, prompt) {
  
// }

// Controller Function
async function handleCropMonitoring(req, res) {
  try {
    const { cropName,inputText } = req.body;

    if (!inputText ) {
      return res.status(400).json({ error: "No input text provided." });
    }

    // Prepare the prompt
    const prompt = `
    You are an expert crop growth analyst.
    
    Analyze the following environmental data for a crop named "${cropName}". 
    The input data might be in JSON or XML format. Explain how well the conditions match this crop's ideal growth requirements.
    Also provide improvement suggestions if needed.
    
    Environmental Data:
    ${inputText}
    `;

    // const result = await analyzeText(inputText, prompt);

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

  const result = await model.generateContent(prompt);
  const response = await result.response;
  const text = response.text;


  return res.status(200).json({ result:text});

    // res.json({ result });
  } catch (err) {
    console.error("Error analyzing crop monitoring data:", err);
    res.status(500).json({ error: "Failed to analyze crop monitoring data." });
  }
}

module.exports = { handleCropMonitoring };
