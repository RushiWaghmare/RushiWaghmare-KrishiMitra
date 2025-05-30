
import { useState } from "react";
import { useLanguage } from "../contexts/LanguageContext";

export type CropType = {
  id: number;
  name: string;
  scientificName: string;
  imageUrl: string;
  growingSeason: string;
  waterRequirements: string;
  soilType: string;
  averageYield: string;
  description: string;
};

const CropCard = ({ crop }: { crop: CropType }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const { t } = useLanguage();

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
      <img
        src={crop.imageUrl}
        alt={crop.name}
        className="w-full h-48 object-cover"
      />
      <div className="p-4">
        <h3 className="text-xl font-bold text-farm-green mb-1">{crop.name}</h3>
        <p className="text-sm text-gray-500 italic mb-4">{crop.scientificName}</p>
        
        <div className="grid grid-cols-2 gap-2 mb-4">
          <div>
            <span className="text-sm font-semibold text-gray-600">{t("crop.season")}</span>
            <p className="text-sm">{crop.growingSeason}</p>
          </div>
          <div>
            <span className="text-sm font-semibold text-gray-600">{t("crop.water")}</span>
            <p className="text-sm">{crop.waterRequirements}</p>
          </div>
          <div>
            <span className="text-sm font-semibold text-gray-600">{t("crop.soil")}</span>
            <p className="text-sm">{crop.soilType}</p>
          </div>
          <div>
            <span className="text-sm font-semibold text-gray-600">{t("crop.yield")}</span>
            <p className="text-sm">{crop.averageYield}</p>
          </div>
        </div>
        
        <div className={`overflow-hidden transition-all duration-300 ${isExpanded ? "max-h-96" : "max-h-0"}`}>
          <p className="text-gray-700 mb-4">{crop.description}</p>
        </div>
        
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full bg-farm-green-light text-white py-2 rounded-md hover:bg-farm-green transition"
        >
          {isExpanded ? t("crop.showLess") : t("crop.showMore")}
        </button>
      </div>
    </div>
  );
};

export default CropCard;
