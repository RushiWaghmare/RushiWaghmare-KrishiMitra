
import { useState } from "react";
import Footer from "../components/Footer";
import CropCard, { CropType } from "../components/CropCard";
import { useLanguage } from "../contexts/LanguageContext";


// Sample crop data with translations
const cropsData: {
  id: number;
  name: { en: string; mr: string };
  scientificName: string;
  imageUrl: string;
  growingSeason: { en: string; mr: string };
  waterRequirements: { en: string; mr: string };
  soilType: { en: string; mr: string };
  averageYield: { en: string; mr: string };
  description: { en: string; mr: string };
}[] = [
  {
    id: 1,
    name: {
      en: "Rice (Paddy)",
      mr: "तांदूळ (भात)"
    },
    scientificName: "Oryza sativa",
    imageUrl: "/public/images/images/rice.jpeg",
    growingSeason: {
      en: "Kharif (Monsoon)",
      mr: "खरीप (पावसाळा)"
    },
    waterRequirements: {
      en: "High",
      mr: "जास्त"
    },
    soilType: {
      en: "Clay or Clay Loam",
      mr: "चिकणमाती किंवा चिकणमाती मिश्रित"
    },
    averageYield: {
      en: "3-6 tons/hectare",
      mr: "३-६ टन/हेक्टर"
    },
    description: {
      en: "Rice is the staple food grain for more than half of the world's population. It is grown in flooded fields (paddy) and requires warm temperatures and high rainfall. In India, it's primarily a Kharif crop planted in June-July and harvested in November-December.",
      mr: "तांदूळ हे जगातील अर्ध्याहून अधिक लोकसंख्येचे प्रमुख अन्नधान्य पीक आहे. हे पाणथळ शेतात (भातशेती) लावले जाते आणि त्यासाठी उष्ण तापमान आणि अधिक पावसाची आवश्यकता असते. भारतात, हे प्रामुख्याने जून-जुलैमध्ये लागवड केले जाणारे आणि नोव्हेंबर-डिसेंबरमध्ये कापणी केले जाणारे खरीप पीक आहे."
    }
  },
  {
    id: 2,
    name: {
      en: "Wheat",
      mr: "गहू"
    },
    scientificName: "Triticum aestivum",
    imageUrl: "/public/images/images/wheat.jpg",
    growingSeason: {
      en: "Rabi (Winter)",
      mr: "रबी (हिवाळा)"
    },
    waterRequirements: {
      en: "Medium",
      mr: "मध्यम"
    },
    soilType: {
      en: "Loam or Clay Loam",
      mr: "चिकनमाती किंवा चिकणमाती मिश्रीत"
    },
    averageYield: {
      en: "3-5 tons/hectare",
      mr: "३-५ टन/हेक्टर"
    },
    description: {
      en: "Wheat is the second most important cereal crop in India after rice. It is a Rabi crop sown in October-December and harvested in April-May. It requires a cold growing season and bright sunshine at the time of ripening.",
      mr: "गहू हे भारतातील तांदळानंतरचे दुसरे सर्वात महत्त्वाचे धान्य पीक आहे. हे ऑक्टोबर-डिसेंबरमध्ये पेरले जाणारे आणि एप्रिल-मेमध्ये कापणी केले जाणारे रबी पीक आहे. परिपक्व होण्याच्या वेळी थंड वाढीचा हंगाम आणि उज्ज्वल सूर्यप्रकाशाची आवश्यकता असते."
    }
  },
  {
    id: 3,
    name: {
      en: "Corn (Maize)",
      mr: "मका"
    },
    scientificName: "Zea mays",
    imageUrl: "/public/images/images/corn.jpeg",
    growingSeason: {
      en: "Kharif and Rabi",
      mr: "खरीप आणि रबी"
    },
    waterRequirements: {
      en: "Medium",
      mr: "मध्यम"
    },
    soilType: {
      en: "Well-drained Loamy Soil",
      mr: "चांगल्या निचऱ्याची चिकट माती"
    },
    averageYield: {
      en: "2.5-4 tons/hectare",
      mr: "२.५-४ टन/हेक्टर"
    },
    description: {
      en: "Maize is cultivated throughout the year in India. It is primarily a Kharif crop with 85% of the area under cultivation in the rainy season. Maize is a versatile crop that can be grown in various climates.",
      mr: "मका भारतात वर्षभर लागवड केली जाते. पावसाळ्याच्या हंगामात ८५% क्षेत्रावर लागवड केली जाणारी ही प्रामुख्याने खरीप पीक आहे. मका हे विविध वातावरणात वाढविले जाऊ शकणारे बहुमुखी पीक आहे."
    }
  },
  {
    id: 4,
    name: {
      en: "Cotton",
      mr: "कापूस"
    },
    scientificName: "Gossypium hirsutum",
    imageUrl: "/images/images/cotton.jpeg",
    growingSeason: {
      en: "Kharif (Monsoon)",
      mr: "खरीप (पावसाळा)"
    },
    waterRequirements: {
      en: "Medium",
      mr: "मध्यम"
    },
    soilType: {
      en: "Black Soil, Alluvial Soil",
      mr: "काळी जमीन, गाळाची माती"
    },
    averageYield: {
      en: "2-3 bales/hectare",
      mr: "२-३ गाठी/हेक्टर"
    },
    description: {
      en: "Cotton is one of the major cash crops in India and plays a significant role in the country's agricultural and industrial economy. It provides livelihoods to millions of farmers and workers in the textile sector.",
      mr: "कापूस हे भारतातील प्रमुख रोख पिकांपैकी एक आहे आणि देशाच्या कृषी आणि औद्योगिक अर्थव्यवस्थेत महत्त्वाची भूमिका बजावते. हे लाखो शेतकरी आणि वस्त्रोद्योग क्षेत्रातील कामगारांना उपजीविका प्रदान करते."
    }
  },
  {
    id: 5,
    name: {
      en: "Mustard",
      mr: "मोहरी"
    },
    scientificName: "Brassica juncea",
    imageUrl: "/images/images/mustard.jpg",
    growingSeason: {
      en: "Rabi (Winter)",
      mr: "रबी (हिवाळा)"
    },
    waterRequirements: {
      en: "Low to Medium",
      mr: "कमी ते मध्यम"
    },
    soilType: {
      en: "Loamy, Sandy Loam",
      mr: "चिकट मातीयुक्त, वाळूमिश्रित"
    },
    averageYield: {
      en: "1-1.5 tons/hectare",
      mr: "१-१.५ टन/हेक्टर"
    },
    description: {
      en: "Mustard is an important oilseed crop in India. It is grown in the Rabi season and is a good choice for farmers in regions with limited water availability. The crop has a short duration and can be grown as a catch crop.",
      mr: "मोहरी हे भारतातील महत्वाचे तेलबिया पीक आहे. हे रबी हंगामात वाढविले जाते आणि मर्यादित पाणी उपलब्धता असलेल्या प्रदेशातील शेतकऱ्यांसाठी चांगला पर्याय आहे. या पिकाचा कालावधी कमी असतो आणि कॅच क्रॉप म्हणून वाढवला जाऊ शकतो."
    }
  },
  {
    id: 6,
    name: {
      en: "Sugarcane",
      mr: "ऊस"
    },
    scientificName: "Saccharum officinarum",
    imageUrl: "/images/images/sugarcane.jpg",
    growingSeason: {
      en: "Year-round (12-18 months)",
      mr: "वर्षभर (१२-१८ महिने)"
    },
    waterRequirements: {
      en: "High",
      mr: "जास्त"
    },
    soilType: {
      en: "Deep, Well-drained Loamy Soil",
      mr: "खोल, चांगल्या निचऱ्याची चिकट माती"
    },
    averageYield: {
      en: "70-100 tons/hectare",
      mr: "७०-१०० टन/हेक्टर"
    },
    description: {
      en: "Sugarcane is a long-duration crop that takes 12-18 months to mature. It is the main source of sugar in India and is also used for producing jaggery, molasses, and ethanol.",
      mr: "ऊस हे परिपक्व होण्यास १२-१८ महिने लागणारे दीर्घकालीन पीक आहे. हा भारतातील साखरेचा मुख्य स्रोत आहे आणि गूळ, मळी आणि इथेनॉल तयार करण्यासाठीही वापरला जातो."
    }
  }
];

const Crops = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterSeason, setFilterSeason] = useState("");
  const { language, t } = useLanguage();
  
  // Transform the data to the current language
  const transformedCropsData: CropType[] = cropsData.map(crop => ({
    id: crop.id,
    name: crop.name[language],
    scientificName: crop.scientificName,
    imageUrl: crop.imageUrl,
    growingSeason: crop.growingSeason[language],
    waterRequirements: crop.waterRequirements[language],
    soilType: crop.soilType[language],
    averageYield: crop.averageYield[language],
    description: crop.description[language]
  }));
  
  const filteredCrops = transformedCropsData.filter((crop) => {
    const matchesSearch = crop.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      crop.scientificName.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesSeason = filterSeason === "" || crop.growingSeason.includes(filterSeason);
    
    return matchesSearch && matchesSeason;
  });

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex-grow">
        {/* Header */}
        <div className="bg-farm-green text-white py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-4xl font-bold mb-4">{t("crops.title")}</h1>
            <p className="text-xl max-w-2xl">
              {t("crops.description")}
            </p>
          </div>
        </div>
        
        {/* Filters */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="bg-white shadow-md rounded-lg p-6 mb-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="col-span-1 md:col-span-2">
                <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-1">
                  {t("crops.search")}
                </label>
                <input
                  type="text"
                  id="search"
                  placeholder={t("crops.searchPlaceholder")}
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full border border-gray-300 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-farm-green focus:border-transparent"
                />
              </div>
              
              <div>
                <label htmlFor="season" className="block text-sm font-medium text-gray-700 mb-1">
                  {t("crops.filterSeason")}
                </label>
                <select
                  id="season"
                  value={filterSeason}
                  onChange={(e) => setFilterSeason(e.target.value)}
                  className="w-full border border-gray-300 rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-farm-green focus:border-transparent"
                >
                  <option value="">{t("crops.allSeasons")}</option>
                  <option value={language === "en" ? "Kharif" : "खरीप"}>{t("crops.kharif")}</option>
                  <option value={language === "en" ? "Rabi" : "रबी"}>{t("crops.rabi")}</option>
                  <option value={language === "en" ? "Zaid" : "झाईड"}>{t("crops.zaid")}</option>
                </select>
              </div>
            </div>
          </div>
          
          {/* Crop Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {filteredCrops.length > 0 ? (
              filteredCrops.map((crop) => (
                <CropCard key={crop.id} crop={crop} />
              ))
            ) : (
              <div className="col-span-full text-center py-8">
                <p className="text-gray-500 text-lg">{t("crops.noResults")}</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default Crops;
