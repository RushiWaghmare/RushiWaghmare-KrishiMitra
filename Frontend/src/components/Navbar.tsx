
import { useState } from "react";
import { Link } from "react-router-dom";
import { Menu, X, Leaf, FileSearch, Beaker } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { language, setLanguage, t } = useLanguage();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'mr' : 'en');
  };

  return (
    <nav className="bg-gradient-to-r from-farm-green-dark to-farm-green shadow-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="flex items-center">
                <span className="text-white font-bold text-xl">KrishiMitra</span>
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link to="/" className="inline-flex items-center px-1 pt-1 border-b-2 border-white text-sm font-medium text-white hover:text-farm-beige transition-colors">
                {t("nav.home")}
              </Link>
              <Link to="/crops" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-white hover:text-farm-beige hover:border-farm-beige transition-colors">
                {t("nav.cropDatabase")}
              </Link>
              <Link to="/schemes" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-white hover:text-farm-beige hover:border-farm-beige transition-colors">
                {t("nav.govtSchemes")}
              </Link>
              <Link to="/fertilizers" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-white hover:text-farm-beige hover:border-farm-beige transition-colors">
                <Beaker className="mr-1 h-4 w-4" />
                {t("nav.fertilizers")}
              </Link>
              <Link to="/disease" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-white hover:text-farm-beige hover:border-farm-beige transition-colors">
                <Leaf className="mr-1 h-4 w-4" />
                {t("nav.diseaseDetection")}
              </Link>
              <a 
                href="https://crop-monitoring.streamlit.app/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-white hover:text-farm-beige hover:border-farm-beige transition-colors"
              >
                 {t("nav.cropMonitoring")}
              </a>
              
              <Link to="/croppriceprediction" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-white hover:text-farm-beige hover:border-farm-beige transition-colors">
                <Beaker className="mr-1 h-4 w-4" />
                {t("nav.cropPricePrediction")}
              </Link>
            </div>
          </div>
          <div className="flex items-center">
            <button 
              onClick={toggleLanguage}
              className="mr-4 bg-white/10 hover:bg-white/20 text-white px-3 py-1 rounded-md transition-colors"
            >
              {language === 'en' ? 'मराठी' : 'English'}
            </button>
            <div className="-mr-2 flex items-center sm:hidden">
              <button
                onClick={toggleMenu}
                className="inline-flex items-center justify-center p-2 rounded-md text-white hover:text-farm-beige hover:bg-farm-green-light focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                aria-expanded="false"
              >
                <span className="sr-only">Open main menu</span>
                {isMenuOpen ? <X className="block h-6 w-6" /> : <Menu className="block h-6 w-6" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {isMenuOpen && (
        <div className="sm:hidden bg-farm-green-dark">
          <div className="pt-2 pb-3 space-y-1">
            <Link to="/" className="block pl-3 pr-4 py-2 border-l-4 border-white text-base font-medium text-white bg-farm-green-light/30">
              {t("nav.home")}
            </Link>
            <Link to="/crops" className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors">
              {t("nav.cropDatabase")}
            </Link>
            <Link to="/schemes" className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors">
              {t("nav.govtSchemes")}
            </Link>
            <Link to="/fertilizers" className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors">
              <div className="flex items-center">
                <Beaker className="mr-2 h-4 w-4" />
                {t("nav.fertilizers")}
              </div>
            </Link>
            <Link to="/disease" className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors">
              <div className="flex items-center">
                <Leaf className="mr-2 h-4 w-4" />
                {t("nav.diseaseDetection")}
              </div>
            </Link>
            <a 
              href="https://crop-monitoring.streamlit.app/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors"
              >
                Crop Monitoring
            </a>

            <Link to="/croppriceprediction" className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors">
              <div className="flex items-center">
                <FileSearch className="mr-2 h-4 w-4" />
                {t("nav.cropPricePrediction")}
              </div>
            </Link>

            

            
            
            <button 
              onClick={toggleLanguage}
              className="block w-full text-left pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-white hover:bg-farm-green-light/20 hover:border-farm-beige transition-colors"
            >
              {language === 'en' ? 'मराठी' : 'English'}
            </button>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
