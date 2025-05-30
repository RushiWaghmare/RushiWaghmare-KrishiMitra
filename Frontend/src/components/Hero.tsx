
import { Link } from "react-router-dom";
import { useLanguage } from "../contexts/LanguageContext";

const Hero = () => {
  const { t } = useLanguage();
  
  return (
    <div className="relative">
      {/* Background image with overlay */}
      <div className="absolute inset-0">
        <img
          src="/lovable-uploads/9526d9c1-b91f-444a-866f-ed7da6b7823d.png"
          alt="Farmer background"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-black/60"></div>
      </div>

      {/* Content on top of the image */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 md:py-32">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold mb-6 text-white">
              {t("hero.title")}
            </h1>
            <p className="text-xl mb-8 text-white">
              {t("hero.description")}
            </p>
            <div className="flex flex-wrap gap-4">
              {/* <Link
                to="/crops"
                className="bg-white text-farm-green font-bold py-3 px-6 rounded-md hover:bg-farm-beige transition duration-300"
              >
                {t("hero.button.kisanBazaar")}
              </Link> */}
              <a 
                href="https://kisan-bazaar-india.vercel.app/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="bg-white text-farm-green font-bold py-3 px-6 rounded-md hover:bg-farm-beige transition duration-300"
              >
                {t("hero.button.kisanBazaar")}
              </a>
              <Link
                to="/schemes"
                className="bg-transparent border-2 border-white text-white font-bold py-3 px-6 rounded-md hover:bg-white/10 transition duration-300"
              >
                {t("hero.button.viewSchemes")}
              </Link>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default Hero;
