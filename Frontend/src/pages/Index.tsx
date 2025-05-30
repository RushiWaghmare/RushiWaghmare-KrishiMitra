
import Footer from "../components/Footer";
import Hero from "../components/Hero";
import Weather from "../components/Weather";
import SeasonalGuide from "../components/SeasonalGuide";
import { Link } from "react-router-dom";
import { Calendar, Leaf } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

const Index = () => {
  const { t } = useLanguage();
  
  return (
    <div className="min-h-screen flex flex-col">
      {/* Main Content */}
      <div className="flex-grow">
        {/* Hero Section - Full Width */}
        <Hero />
        
        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          {/* Features Section */}
          <div className="mb-16">
            <h2 className="text-3xl font-bold text-center text-gray-900 mb-8">
              {t("features.heading")}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <div className="w-12 h-12 bg-farm-green-light/20 rounded-full flex items-center justify-center mb-4">
                  <Leaf className="h-6 w-6 text-farm-green" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{t("features.cropInfo.title")}</h3>
                <p className="text-gray-600 mb-4">
                  {t("features.cropInfo.description")}
                </p>
                <Link
                  to="/crops"
                  className="text-farm-green hover:text-farm-green-dark font-medium"
                >
                  {t("features.cropInfo.link")}
                </Link>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-md">
                <div className="w-12 h-12 bg-farm-green-light/20 rounded-full flex items-center justify-center mb-4">
                  <Calendar className="h-6 w-6 text-farm-green" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{t("features.seasonalGuides.title")}</h3>
                <p className="text-gray-600 mb-4">
                  {t("features.seasonalGuides.description")}
                </p>
                <a
                  href="#seasonal-guide"
                  className="text-farm-green hover:text-farm-green-dark font-medium"
                >
                  {t("features.seasonalGuides.link")}
                </a>
              </div>
              
              <div className="bg-white p-6 rounded-lg shadow-md">
                <div className="w-12 h-12 bg-farm-green-light/20 rounded-full flex items-center justify-center mb-4">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 text-farm-green"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                    />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{t("features.govtSchemes.title")}</h3>
                <p className="text-gray-600 mb-4">
                  {t("features.govtSchemes.description")}
                </p>
                <Link
                  to="/schemes"
                  className="text-farm-green hover:text-farm-green-dark font-medium"
                >
                  {t("features.govtSchemes.link")}
                </Link>
              </div>
            </div>
          </div>
          
          {/* Weather and Seasonal Guide */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
            <Weather />
            <div id="seasonal-guide">
              <SeasonalGuide />
            </div>
          </div>
          
          {/* CTA Section */}
          <div className="bg-farm-green-light/10 rounded-lg p-8 text-center">
            <h2 className="text-2xl font-bold text-farm-green mb-4">
              {t("cta.title")}
            </h2>
            <p className="text-gray-700 mb-6 max-w-2xl mx-auto">
              {t("cta.description")}
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                to="/crops"
                className="bg-farm-green text-white font-bold py-3 px-6 rounded-md hover:bg-farm-green-dark transition duration-300"
              >
                {t("cta.button.browseCrops")}
              </Link>
              <Link
                to="/schemes"
                className="bg-white text-farm-green border border-farm-green font-bold py-3 px-6 rounded-md hover:bg-farm-green-light/10 transition duration-300"
              >
                {t("cta.button.viewSchemes")}
              </Link>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default Index;
