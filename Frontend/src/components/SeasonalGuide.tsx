
import { useState } from "react";
import { useLanguage } from "../contexts/LanguageContext";

type Season = "kharif" | "rabi" | "zaid";

const SeasonalGuide = () => {
  const [activeSeason, setActiveSeason] = useState<Season>("kharif");
  const { t } = useLanguage();
  
  const seasonalData = {
    kharif: {
      name: t("seasonal.kharif"),
      months: "June to October",
      description: t("seasonal.kharif.description"),
      crops: ["Rice", "Maize", "Jowar", "Bajra", "Tur", "Moong", "Urad", "Cotton", "Jute", "Groundnut", "Soybean"]
    },
    rabi: {
      name: t("seasonal.rabi"),
      months: "October to March",
      description: t("seasonal.rabi.description"),
      crops: ["Wheat", "Barley", "Gram", "Peas", "Mustard", "Linseed"]
    },
    zaid: {
      name: t("seasonal.zaid"),
      months: "March to June",
      description: t("seasonal.zaid.description"),
      crops: ["Cucumber", "Pumpkin", "Bitter Gourd", "Watermelon", "Muskmelon", "Gourds"]
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="bg-farm-green text-white p-4">
        <h3 className="font-bold text-xl">{t("seasonal.title")}</h3>
      </div>
      
      <div className="bg-farm-green-light/10 p-4">
        <div className="flex border-b">
          {(Object.keys(seasonalData) as Season[]).map((season) => (
            <button
              key={season}
              onClick={() => setActiveSeason(season)}
              className={`px-4 py-2 font-medium text-sm ${
                activeSeason === season
                  ? "border-b-2 border-farm-green text-farm-green"
                  : "text-gray-600 hover:text-farm-green"
              }`}
            >
              {seasonalData[season].name}
            </button>
          ))}
        </div>
      </div>
      
      <div className="p-4">
        <div className="mb-4">
          <p className="text-sm text-gray-600 mb-1">
            <span className="font-semibold">{t("seasonal.months")}</span> {seasonalData[activeSeason].months}
          </p>
          <p className="text-gray-700">
            {seasonalData[activeSeason].description}
          </p>
        </div>
        
        <div>
          <h4 className="font-semibold text-gray-700 mb-2">{t("seasonal.recommendedCrops")}</h4>
          <div className="flex flex-wrap gap-2">
            {seasonalData[activeSeason].crops.map((crop) => (
              <span
                key={crop}
                className="bg-farm-green-light/10 text-farm-green-dark px-2 py-1 rounded-full text-sm"
              >
                {crop}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SeasonalGuide;
