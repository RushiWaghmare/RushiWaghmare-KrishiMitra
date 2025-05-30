import React, { useState } from "react";
import axios from "axios";

import ReactMarkdown from "react-markdown";
import { Leaf } from "lucide-react";
import { useLanguage } from '../contexts/LanguageContext'; 

const DiseaseDetection = () => {
  const { t } = useLanguage();
  const [image, setImage] = useState<File | null>(null);
  const [language, setLanguage] = useState("en");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleSubmit = async () => {
    if (!image) return alert("Please select an image");
    const formData = new FormData();
    formData.append("image", image);
    formData.append("language", language);

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:5000/disease", formData);
      setResult(res.data.result);
    } catch (err: any) {
      setResult("Error: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

 

  return (
    <div className="min-h-screen bg-[#f7f7e8]">
      {/* Main Content Banner */}
      <div className="bg-[#2C684E] text-white py-16 px-6">
        <div className="container mx-auto">
          <h1 className="text-4xl font-bold mb-4">{t("disease.title")}</h1>
          <p className="text-xl max-w-3xl">
          {t("disease.description")}
          </p>
        </div>
      </div>

      {/* Analysis Tool */}
      <div className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            {/* Title Bar */}
            <div className="bg-[#2C684E] text-white py-4 px-6">
              <h2 className="text-xl font-semibold"> {t("disease.heading")}</h2>
            </div>

            <div className="p-8">
              <div className="mb-6 flex flex-col md:flex-row gap-6">
                <div className="flex-1">
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">{t("disease.imageUpload")}</label>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:bg-gray-50 transition-colors cursor-pointer">
                      <input 
                        type="file" 
                        accept="image/*" 
                        onChange={handleImageChange} 
                        className="hidden" 
                        id="plant-image" 
                      />
                      <label htmlFor="plant-image" className="cursor-pointer flex flex-col items-center justify-center py-4">
                        <Leaf className="h-12 w-12 text-[#2C684E] mb-2" />
                        <span className="text-gray-600">{t("disease.drag")}</span>
                        <span className="text-sm text-gray-500 mt-1">{t("disease.span")}</span>
                      </label>
                    </div>
                  </div>
                  
                  <div className="mb-4">
                    <label className="block text-gray-700 font-medium mb-2">{t("disease.toggle")}</label>
                    <select 
                      value={language} 
                      onChange={(e) => setLanguage(e.target.value)} 
                      className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="en">English</option>
                      <option value="mr">मराठी</option>
                    </select>
                  </div>
                  
                  <button 
                    onClick={handleSubmit} 
                    className="w-full py-3 bg-[#2C684E] text-white rounded-md font-medium hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={!image || loading}
                  >
                    {loading ? t("disease.loading") : t("disease.button")}
                  </button>
                </div>
                
                {/* Image Preview */}
                <div className="flex-1">
                  {previewUrl ? (
                    <div className="aspect-square rounded-lg border border-gray-200 overflow-hidden flex items-center justify-center bg-gray-50">
                      <img 
                        src={previewUrl} 
                        alt="Plant preview" 
                        className="max-w-full max-h-full object-contain" 
                      />
                    </div>
                  ) : (
                    <div className="aspect-square rounded-lg border border-gray-200 flex items-center justify-center bg-gray-50">
                      <p className="text-gray-400 text-center px-4">
                      {t("disease.imagePreview")}
                      </p>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Results Section */}
              {(loading || result) && (
                <div className="mt-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
                  <h2 className="text-xl font-semibold mb-4">{t("disease.result")}</h2>
                  
                  {loading && (
                    <div className="flex items-center justify-center py-8">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#2C684E]"></div>
                    </div>
                  )}
                  
                  {result && (
                    <>
                      <div className="bg-white p-4 rounded-md border border-gray-200 mb-4 whitespace-pre-wrap">
                        <ReactMarkdown>{result}</ReactMarkdown>
                      </div>
                      
                      
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiseaseDetection;
